"""
Filter utilities for evaluating filter expressions against chunks.

Supports SQL-like filter syntax including:
- Comparison operators: =, !=, <, <=, >, >=
- LIKE operator with wildcard support (%, _)
- Logical operators: AND, OR, NOT
- String literals with single or double quotes
- Numeric literals
- Metadata field access with dot notation

Examples:
- color LIKE "red%"
- likes > 50
- metadata.category = "news" AND metadata.score >= 0.8
- text LIKE "%machine learning%" OR metadata.tags LIKE "%AI%"
"""

import re
from typing import Any, List
import operator


class FilterError(Exception):
    pass


class FilterOperator:
    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    LIKE = "LIKE"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class FilterToken:
    def __init__(self, type_: str, value: str, position: int = 0):
        self.type = type_
        self.value = value
        self.position = position

    def __repr__(self):
        return f"FilterToken({self.type}, {self.value!r})"


class FilterLexer:
    TOKEN_PATTERNS = [
        ("QUOTED_STRING", r'"([^"\\]|\\.)*"|\'([^\'\\]|\\.)*\''),
        ("NUMBER", r"\d+(\.\d*)?|\.\d+"),
        ("LIKE", r"\bLIKE\b"),
        ("AND", r"\bAND\b"),
        ("OR", r"\bOR\b"),
        ("NOT", r"\bNOT\b"),
        ("OPERATOR", r">=|<=|!=|>|<|="),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_.]*"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("WHITESPACE", r"\s+"),
    ]

    def __init__(self):
        self.pattern = "|".join(
            f"(?P<{name}>{pattern})" for name, pattern in self.TOKEN_PATTERNS
        )
        self.regex = re.compile(self.pattern, re.IGNORECASE)

    def tokenize(self, text: str) -> List[FilterToken]:
        tokens = []
        pos = 0

        for match in self.regex.finditer(text):
            token_type = match.lastgroup
            token_value = match.group()

            if token_type == "WHITESPACE":
                continue

            if token_type == "QUOTED_STRING":
                token_value = token_value[1:-1].replace('\\"', '"').replace("\\'", "'")
                token_type = "STRING"

            tokens.append(FilterToken(token_type, token_value, match.start()))
            pos = match.end()

        if pos < len(text):
            raise FilterError(f"Unexpected character at position {pos}: '{text[pos]}'")

        return tokens


class FilterExpression:
    def evaluate(self, chunk) -> bool:
        raise NotImplementedError


class ComparisonExpression(FilterExpression):
    def __init__(self, field: str, operator: str, value: Any):
        self.field = field
        self.operator = operator
        self.value = value

    def _get_field_value(self, chunk, field: str) -> Any:
        if field == "text":
            return chunk.text
        elif field == "id":
            return chunk.id
        elif field == "document_id":
            return chunk.document_id
        elif field == "created_at":
            return chunk.created_at
        elif field.startswith("metadata."):
            metadata_key = field[9:]
            return chunk.metadata.get(metadata_key)
        elif field in chunk.metadata:
            return chunk.metadata.get(field)
        else:
            return None

    def _compare_like(self, text: str, pattern: str) -> bool:
        if text is None:
            return False

        text = str(text).lower()
        pattern = pattern.lower()

        regex_pattern = pattern.replace("%", ".*").replace("_", ".")
        regex_pattern = f"^{regex_pattern}$"

        return bool(re.match(regex_pattern, text))

    def evaluate(self, chunk) -> bool:
        field_value = self._get_field_value(chunk, self.field)

        if self.operator == FilterOperator.LIKE:
            return self._compare_like(field_value, self.value)

        if field_value is None:
            return self.operator == FilterOperator.NE and self.value is not None

        try:
            if isinstance(self.value, (int, float)) and isinstance(field_value, str):
                field_value = float(field_value)
            elif isinstance(self.value, str) and isinstance(field_value, (int, float)):
                field_value = str(field_value)
        except (ValueError, TypeError):
            pass

        ops = {
            FilterOperator.EQ: operator.eq,
            FilterOperator.NE: operator.ne,
            FilterOperator.LT: operator.lt,
            FilterOperator.LE: operator.le,
            FilterOperator.GT: operator.gt,
            FilterOperator.GE: operator.ge,
        }

        try:
            return ops[self.operator](field_value, self.value)
        except (TypeError, ValueError):
            return False


class LogicalExpression(FilterExpression):
    def __init__(
        self, operator: str, left: FilterExpression, right: FilterExpression = None
    ):
        self.operator = operator
        self.left = left
        self.right = right

    def evaluate(self, chunk) -> bool:
        if self.operator == FilterOperator.NOT:
            return not self.left.evaluate(chunk)
        elif self.operator == FilterOperator.AND:
            return self.left.evaluate(chunk) and self.right.evaluate(chunk)
        elif self.operator == FilterOperator.OR:
            return self.left.evaluate(chunk) or self.right.evaluate(chunk)
        else:
            raise FilterError(f"Unknown logical operator: {self.operator}")


class FilterParser:
    def __init__(self, tokens: List[FilterToken]):
        self.tokens = tokens
        self.position = 0

    def current_token(self) -> FilterToken:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def consume(self, expected_type: str = None) -> FilterToken:
        token = self.current_token()
        if token is None:
            raise FilterError("Unexpected end of expression")

        if expected_type and token.type != expected_type:
            raise FilterError(f"Expected {expected_type}, got {token.type}")

        self.position += 1
        return token

    def parse(self) -> FilterExpression:
        expr = self.parse_or_expression()

        if self.position < len(self.tokens):
            raise FilterError(f"Unexpected token: {self.current_token()}")

        return expr

    def parse_or_expression(self) -> FilterExpression:
        expr = self.parse_and_expression()

        while self.current_token() and self.current_token().type == "OR":
            self.consume("OR")
            right = self.parse_and_expression()
            expr = LogicalExpression(FilterOperator.OR, expr, right)

        return expr

    def parse_and_expression(self) -> FilterExpression:
        expr = self.parse_not_expression()

        while self.current_token() and self.current_token().type == "AND":
            self.consume("AND")
            right = self.parse_not_expression()
            expr = LogicalExpression(FilterOperator.AND, expr, right)

        return expr

    def parse_not_expression(self) -> FilterExpression:
        if self.current_token() and self.current_token().type == "NOT":
            self.consume("NOT")
            expr = self.parse_primary_expression()
            return LogicalExpression(FilterOperator.NOT, expr)

        return self.parse_primary_expression()

    def parse_primary_expression(self) -> FilterExpression:
        token = self.current_token()

        if token is None:
            raise FilterError("Unexpected end of expression")

        if token.type == "LPAREN":
            self.consume("LPAREN")
            expr = self.parse_or_expression()
            self.consume("RPAREN")
            return expr

        field_token = self.consume("IDENTIFIER")
        field = field_token.value

        operator_token = self.current_token()
        if operator_token is None:
            raise FilterError("Expected operator after field name")

        if operator_token.type in ["OPERATOR", "LIKE"]:
            operator_str = self.consume().value
        else:
            raise FilterError(f"Expected operator, got {operator_token.type}")

        value_token = self.current_token()
        if value_token is None:
            raise FilterError("Expected value after operator")

        if value_token.type == "STRING":
            value = self.consume("STRING").value
        elif value_token.type == "NUMBER":
            value_str = self.consume("NUMBER").value
            value = float(value_str) if "." in value_str else int(value_str)
        else:
            raise FilterError(f"Expected value, got {value_token.type}")

        return ComparisonExpression(field, operator_str.upper(), value)


class ChunkFilter:
    def __init__(self, filter_string: str):
        self.filter_string = filter_string
        self.expression = self._parse_filter_string(filter_string)

    def _parse_filter_string(self, filter_string: str) -> FilterExpression:
        if not filter_string or not filter_string.strip():
            raise FilterError("Filter string cannot be empty")

        lexer = FilterLexer()
        tokens = lexer.tokenize(filter_string)

        if not tokens:
            raise FilterError("Filter string cannot be empty")

        parser = FilterParser(tokens)
        return parser.parse()

    def matches(self, chunk) -> bool:
        try:
            return self.expression.evaluate(chunk)
        except Exception as e:
            raise FilterError(f"Error evaluating filter: {e}")

    def filter_chunks(self, chunks: List) -> List:
        return [chunk for chunk in chunks if self.matches(chunk)]


def check_chunk(chunk, filter_string: str) -> bool:
    if not filter_string or not filter_string.strip():
        return True
    filter_obj = ChunkFilter(filter_string)
    return filter_obj.matches(chunk)
