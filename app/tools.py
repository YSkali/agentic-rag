"""Agent 可调用的工具。

工具 = 一个「模型能通过 function calling 调用的函数」。
用 @tool 装饰器把普通函数包成 LangChain 的 BaseTool：
- docstring 会作为「工具说明」传给模型，模型据此判断「这题要不要调、传什么参数」。
- 这里先放一个自包含的计算器（无外部依赖、离线必跑通），
  后续 Phase B 会用 MCP 协议从「独立进程」发现更多工具，接口完全一样。
"""
import ast
import operator

from langchain_core.tools import tool


# ---------- 安全求值：不用裸 eval ----------
# eval("__import__('os').system('...')") 会执行任意代码，有安全风险。
# 这里用 ast 把字符串解析成「表达式树」，再按白名单逐节点求值，
# 只放行数字、括号、+-*/%** 这些纯算术运算，其余一律拒绝。
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST):
    """按白名单求值 ast 节点，只允许算术表达式。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """计算一个算术表达式，例如 '123*456' 或 '(12+3)*4'。
    支持加减乘除、取余、幂和括号。参数 expression 是待计算的表达式字符串。"""
    if not expression or not expression.strip():
        return "计算出错：表达式不能为空"
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(_safe_eval(tree))
    except Exception as e:  # 语法错 / 白名单外的运算 / 除零，都安全兜底
        return f"计算出错：{e}"


# 模块级工具清单：后续 Phase B 会把 MCP 发现的工具也并进来
TOOLS = [calculator]
