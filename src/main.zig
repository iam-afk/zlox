const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

pub fn main() !void {
    const stdin_file = std.io.getStdIn().reader();
    var br = std.io.bufferedReader(stdin_file);
    const stdin = br.reader();

    const stdout = std.io.getStdOut().writer();

    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = general_purpose_allocator.allocator();

    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    if (args.len > 2) {
        try stdout.print("Usage: zlox [script]\n", .{});
        std.process.exit(64);
    } else if (args.len == 2) {
        try runFile(args[1], stdout, gpa);
    } else {
        try runPrompt(stdin, stdout, gpa);
    }
}

fn runFile(path: []const u8, writer: anytype, allocator: Allocator) !void {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, std.math.maxInt(usize));
    defer allocator.free(bytes);

    var reporter = ErrorReporter{};

    try run(bytes, writer, &reporter, allocator);

    if (reporter.had_errors) {
        std.process.exit(65);
    }
}

fn runPrompt(reader: anytype, writer: anytype, allocator: Allocator) !void {
    var reporter = ErrorReporter{};
    while (true) {
        try writer.print("> ", .{});
        if (try reader.readUntilDelimiterOrEofAlloc(allocator, '\n', std.math.maxInt(usize))) |line| {
            defer allocator.free(line);
            try run(line, writer, &reporter, allocator);
            reporter.reset();
        } else break;
    }
}

fn run(source: []const u8, writer: anytype, reporter: *ErrorReporter, allocator: Allocator) !void {
    var scanner = Scanner.init(source, reporter, allocator);
    defer scanner.deinit();

    try scanner.scanTokens();

    var parser = Parser.init(allocator, scanner.tokens.items, reporter);
    const expression = parser.parse();

    if (reporter.had_errors) return;

    try printAst(writer, expression.?);
    try writer.print("\n", .{});
}

const ErrorReporter = struct {
    had_errors: bool = false,
    fn report(self: *ErrorReporter, line: i32, where: ?Token, message: []const u8) void {
        if (where) |token| switch (token.type) {
            .eof => std.debug.print("[line {}] Error at end: {s}\n", .{ line, message }),
            else => std.debug.print("[line {}] Error '{s}': {s}\n", .{ line, token.lexeme, message }),
        } else std.debug.print("[line {}] Error: {s}\n", .{ line, message });
        self.had_errors = true;
    }
    fn reportToken(self: *ErrorReporter, token: Token, message: []const u8) void {
        self.report(token.line, token, message);
    }
    fn reset(self: *ErrorReporter) void {
        self.had_errors = false;
    }
};

const Token = struct {
    const TypeTag = std.meta.Tag(Type);
    const Type = union(enum) {
        left_paren,
        right_paren,
        left_brace,
        right_brace,
        comma,
        dot,
        minus,
        plus,
        semicolon,
        slash,
        star,
        bang,
        bang_equal,
        equal,
        equal_equal,
        greater,
        greater_equal,
        less,
        less_equal,
        identifier,
        string: []const u8,
        number: f64,
        And,
        Class,
        Else,
        False,
        Fun,
        For,
        If,
        Nil,
        Or,
        Print,
        Return,
        Super,
        This,
        True,
        Var,
        While,
        eof,
    };

    type: Type,
    lexeme: []const u8,
    line: i32,
};

const Scanner = struct {
    source: []const u8,
    reporter: *ErrorReporter,
    tokens: Tokens,
    start: usize = 0,
    current: usize = 0,
    line: i32 = 1,

    const Self = @This();
    const Tokens = ArrayList(Token);
    const Keywords = std.StringHashMapUnmanaged(Token.Type);

    const keywords = std.ComptimeStringMap(Token.Type, .{
        .{ "and", .And },
        .{ "class", .Class },
        .{ "else", .Else },
        .{ "false", .False },
        .{ "for", .For },
        .{ "fun", .Fun },
        .{ "if", .If },
        .{ "nil", .Nil },
        .{ "or", .Or },
        .{ "print", .Print },
        .{ "return", .Return },
        .{ "super", .Super },
        .{ "this", .This },
        .{ "true", .True },
        .{ "var", .Var },
        .{ "while", .While },
    });

    fn init(source: []const u8, reporter: *ErrorReporter, allocator: Allocator) Self {
        return .{
            .source = source,
            .reporter = reporter,
            .tokens = Tokens.init(allocator),
        };
    }

    fn deinit(self: *Self) void {
        self.tokens.deinit();
    }

    fn scanTokens(self: *Self) !void {
        while (!self.isAtEnd()) {
            self.start = self.current;
            try self.scanToken();
        }
        try self.tokens.append(Token{ .type = Token.Type.eof, .lexeme = "", .line = self.line });
    }

    fn isAtEnd(self: *Self) bool {
        return self.current >= self.source.len;
    }

    fn scanToken(self: *Self) !void {
        const c = self.advance();
        switch (c) {
            '(' => try self.addToken(.left_paren),
            ')' => try self.addToken(.right_paren),
            '{' => try self.addToken(.left_brace),
            '}' => try self.addToken(.right_brace),
            ',' => try self.addToken(.comma),
            '.' => try self.addToken(.dot),
            '-' => try self.addToken(.minus),
            '+' => try self.addToken(.plus),
            ';' => try self.addToken(.semicolon),
            '*' => try self.addToken(.star),
            '!' => try self.addToken(if (self.match('=')) .bang_equal else .bang),
            '=' => try self.addToken(if (self.match('=')) .equal_equal else .equal),
            '<' => try self.addToken(if (self.match('=')) .less_equal else .less),
            '>' => try self.addToken(if (self.match('=')) .greater_equal else .greater),
            '/' => if (self.match('/')) {
                while (self.peek() != '\n' and !self.isAtEnd()) _ = self.advance();
            } else try self.addToken(.slash),
            ' ', '\r', '\t' => {},
            '\n' => self.line += 1,
            '"' => try self.string(),
            '0'...'9' => try self.number(),
            'A'...'Z', 'a'...'z', '_' => try self.identifier(),
            else => self.reporter.report(self.line, null, "Unexpected character."),
        }
    }

    fn identifier(self: *Self) !void {
        while (isAlphaNumeric(self.peek())) _ = self.advance();

        const text = self.source[self.start..self.current];
        const token_type = keywords.get(text) orelse .identifier;
        try self.addToken(token_type);
    }

    fn number(self: *Self) !void {
        while (isDigit(self.peek())) _ = self.advance();

        if (self.peek() == '.' and isDigit(self.peekNext())) {
            _ = self.advance();
            while (isDigit(self.peek())) _ = self.advance();
        }

        try self.addToken(Token.Type{ .number = try std.fmt.parseFloat(f64, self.source[self.start..self.current]) });
    }

    fn string(self: *Self) !void {
        while (self.peek() != '"' and !self.isAtEnd()) {
            if (self.peek() == '\n') self.line += 1;
            _ = self.advance();
        }

        if (self.isAtEnd()) {
            self.reporter.report(self.line, null, "Unterminated string.");
            return;
        }

        _ = self.advance();

        const value = self.source[self.start + 1 .. self.current - 1];
        try self.addToken(Token.Type{ .string = value });
    }

    fn match(self: *Self, expected: u8) bool {
        if (self.isAtEnd()) return false;
        if (self.source[self.current] != expected) return false;

        self.current += 1;
        return true;
    }

    fn peek(self: *Self) u8 {
        if (self.isAtEnd()) return 0;
        return self.source[self.current];
    }

    fn peekNext(self: *Self) u8 {
        if (self.current + 1 >= self.source.len) return 0;
        return self.source[self.current + 1];
    }

    fn isAlpha(c: u8) bool {
        return switch (c) {
            'a'...'z', 'A'...'Z', '_' => true,
            else => false,
        };
    }

    fn isAlphaNumeric(c: u8) bool {
        return isAlpha(c) or isDigit(c);
    }

    const isDigit = std.ascii.isDigit;

    fn advance(self: *Self) u8 {
        const c = self.source[self.current];
        self.current += 1;
        return c;
    }

    fn addToken(self: *Self, token_type: Token.Type) !void {
        try self.tokens.append(Token{
            .type = token_type,
            .lexeme = self.source[self.start..self.current],
            .line = self.line,
        });
    }
};

const Expr = union(enum) {
    binary: struct {
        left: *Expr,
        operator: Token,
        right: *Expr,
    },
    grouping: struct {
        expression: *Expr,
    },
    literal: union(enum) {
        string: []const u8,
        number: f64,
        false,
        true,
        nil,
    },
    unary: struct {
        operator: Token,
        right: *Expr,
    },

    fn create(allocator: Allocator, expr: anytype) Allocator.Error!*Expr {
        const e = try allocator.create(Expr);
        e.* = expr;
        return e;
    }
};

fn parenthesize(writer: anytype, name: []const u8, exprs: anytype) !void {
    try std.fmt.format(writer, "({s}", .{name});
    const fields_info = @typeInfo(@TypeOf(exprs)).Struct.fields;
    inline for (fields_info) |field_info| {
        try std.fmt.format(writer, " ", .{});
        try printAst(writer, @field(exprs, field_info.name));
    }
    try std.fmt.format(writer, ")", .{});
}

fn printAst(writer: anytype, expr: *Expr) anyerror!void {
    try switch (expr.*) {
        .binary => |binary| parenthesize(writer, binary.operator.lexeme, .{ binary.left, binary.right }),
        .grouping => |grouping| parenthesize(writer, "group", .{grouping.expression}),
        .literal => |literal| switch (literal) {
            .string => |value| std.fmt.format(writer, "{s}", .{value}),
            .number => |value| std.fmt.format(writer, "{d}", .{value}),
            .false => std.fmt.format(writer, "false", .{}),
            .true => std.fmt.format(writer, "true", .{}),
            .nil => std.fmt.format(writer, "nil", .{}),
        },
        .unary => |unary| parenthesize(writer, unary.operator.lexeme, .{unary.right}),
    };
}

test "a (not very) pretty printer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const a = arena.allocator();

    const expr = try Expr.create(a, .{ .binary = .{
        .left = try Expr.create(a, .{ .unary = .{
            .operator = Token{ .type = .minus, .lexeme = "-", .line = 1 },
            .right = try Expr.create(a, .{ .literal = .{ .number = 123 } }),
        } }),
        .operator = Token{ .type = .star, .lexeme = "*", .line = 1 },
        .right = try Expr.create(a, .{ .grouping = .{
            .expression = try Expr.create(a, .{ .literal = .{ .number = 45.67 } }),
        } }),
    } });

    var buf: [32]u8 = undefined;
    var fbs = std.io.fixedBufferStream(buf[0..]);
    try printAst(fbs.writer(), expr);

    try std.testing.expectEqualSlices(u8, "(* (- 123) (group 45.67))", fbs.getWritten());
}

fn convertToRpn(writer: anytype, name: []const u8, exprs: anytype) !void {
    const fields_info = @typeInfo(@TypeOf(exprs)).Struct.fields;
    inline for (fields_info) |field_info| {
        try printRpn(writer, @field(exprs, field_info.name));
        try std.fmt.format(writer, " ", .{});
    }
    try std.fmt.format(writer, "{s}", .{name});
}

fn printRpn(writer: anytype, expr: *Expr) anyerror!void {
    try switch (expr.*) {
        .binary => |binary| convertToRpn(writer, binary.operator.lexeme, .{ binary.left, binary.right }),
        .grouping => |grouping| printRpn(writer, grouping.expression),
        .literal => |literal| switch (literal) {
            .string => |value| std.fmt.format(writer, "{s}", .{value}),
            .number => |value| std.fmt.format(writer, "{d}", .{value}),
            .false => std.fmt.format(writer, "false", .{}),
            .true => std.fmt.format(writer, "true", .{}),
            .nil => std.fmt.format(writer, "nil", .{}),
        },
        .unary => |unary| convertToRpn(writer, unary.operator.lexeme, .{unary.right}),
    };
}

test "reverse Polish notation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const a = arena.allocator();

    const expr = try Expr.create(a, .{ .binary = .{
        .left = try Expr.create(a, .{ .grouping = .{
            .expression = try Expr.create(a, .{ .binary = .{
                .left = try Expr.create(a, .{ .literal = .{ .number = 1 } }),
                .operator = Token{ .type = .plus, .lexeme = "+", .line = 1 },
                .right = try Expr.create(a, .{ .literal = .{ .number = 2 } }),
            } }),
        } }),
        .operator = Token{ .type = .star, .lexeme = "*", .line = 1 },
        .right = try Expr.create(a, .{ .grouping = .{
            .expression = try Expr.create(a, .{ .binary = .{
                .left = try Expr.create(a, .{ .literal = .{ .number = 4 } }),
                .operator = Token{ .type = .minus, .lexeme = "-", .line = 1 },
                .right = try Expr.create(a, .{ .literal = .{ .number = 3 } }),
            } }),
        } }),
    } });

    var buf: [32]u8 = undefined;
    var fbs = std.io.fixedBufferStream(buf[0..]);
    try printRpn(fbs.writer(), expr);

    try std.testing.expectEqualSlices(u8, "1 2 + 4 3 - *", fbs.getWritten());
}

const Parser = struct {
    allocator: Allocator,
    tokens: []Token,
    reporter: *ErrorReporter,
    current: usize = 0,

    const ParseError = error{Error} || Allocator.Error;

    fn init(allocator: Allocator, tokens: []Token, reporter: *ErrorReporter) Parser {
        return .{ .allocator = allocator, .tokens = tokens, .reporter = reporter };
    }

    fn parse(self: *Parser) ?*Expr {
        return self.expression() catch null;
    }

    fn expression(self: *Parser) !*Expr {
        return self.equality();
    }

    fn equality(self: *Parser) !*Expr {
        var expr = try self.comparison();

        while (self.match(.{ .bang_equal, .equal_equal })) {
            const operator = self.previous();
            const right = try self.comparison();
            expr = try Expr.create(self.allocator, .{ .binary = .{ .left = expr, .operator = operator, .right = right } });
        }

        return expr;
    }

    fn comparison(self: *Parser) !*Expr {
        var expr = try self.term();

        while (self.match(.{ .greater, .greater_equal, .less, .less_equal })) {
            const operator = self.previous();
            const right = try self.term();
            expr = try Expr.create(self.allocator, .{ .binary = .{ .left = expr, .operator = operator, .right = right } });
        }

        return expr;
    }

    fn term(self: *Parser) !*Expr {
        var expr = try self.factor();

        while (self.match(.{ .minus, .plus })) {
            const operator = self.previous();
            const right = try self.factor();
            expr = try Expr.create(self.allocator, .{ .binary = .{ .left = expr, .operator = operator, .right = right } });
        }

        return expr;
    }

    fn factor(self: *Parser) !*Expr {
        var expr = try self.unary();

        while (self.match(.{ .slash, .star })) {
            const operator = self.previous();
            const right = try self.unary();
            expr = try Expr.create(self.allocator, .{ .binary = .{ .left = expr, .operator = operator, .right = right } });
        }

        return expr;
    }

    fn unary(self: *Parser) !*Expr {
        if (self.match(.{ .bang, .minus })) {
            const operator = self.previous();
            const right = try self.unary();
            return try Expr.create(self.allocator, .{ .unary = .{ .operator = operator, .right = right } });
        }

        return self.primary();
    }

    fn primary(self: *Parser) ParseError!*Expr {
        if (self.match(.{.False})) return try Expr.create(self.allocator, .{ .literal = .false });
        if (self.match(.{.True})) return try Expr.create(self.allocator, .{ .literal = .true });
        if (self.match(.{.Nil})) return try Expr.create(self.allocator, .{ .literal = .nil });

        if (self.match(.{ .number, .string })) switch (self.previous().type) {
            .number => |number| return try Expr.create(self.allocator, .{ .literal = .{ .number = number } }),
            .string => |string| return try Expr.create(self.allocator, .{ .literal = .{ .string = string } }),
            else => unreachable,
        };

        if (self.match(.{.left_paren})) {
            const expr = try self.expression();
            _ = try self.consume(.right_paren, "Expect ')' after expression.");
            return try Expr.create(self.allocator, .{ .grouping = .{ .expression = expr } });
        }

        return self.report(self.peek(), "Expect expression.");
    }

    fn match(self: *Parser, types: anytype) bool {
        const fields_info = @typeInfo(@TypeOf(types)).Struct.fields;
        inline for (fields_info) |field_info| {
            if (self.check(@field(types, field_info.name))) {
                _ = self.advance();
                return true;
            }
        }
        return false;
    }

    fn consume(self: *Parser, token_type: Token.Type, message: []const u8) ParseError!Token {
        if (self.check(token_type)) return self.advance();
        return self.report(self.peek(), message);
    }

    fn check(self: *Parser, token_type_tag: Token.TypeTag) bool {
        if (self.isAtEnd()) return false;
        return @as(Token.TypeTag, self.peek().type) == token_type_tag;
    }

    fn advance(self: *Parser) Token {
        if (!self.isAtEnd()) self.current += 1;
        return self.previous();
    }

    fn isAtEnd(self: *Parser) bool {
        return self.peek().type == .eof;
    }

    fn peek(self: *Parser) Token {
        return self.tokens[self.current];
    }

    fn previous(self: *Parser) Token {
        return self.tokens[self.current - 1];
    }

    fn report(self: *Parser, token: Token, message: []const u8) ParseError {
        self.reporter.reportToken(token, message);
        return ParseError.Error;
    }

    fn synchronize(self: *Parser) void {
        self.advance();

        while (!self.isAtEnd()) {
            if (self.previous().type == .semicolon) return;

            switch (self.peek().type) {
                .Class, .Fun, .Var, .For, .If, .While, .Print, .Return => return,
                else => {},
            }

            self.advance();
        }
    }
};
