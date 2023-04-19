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

    for (scanner.tokens.items) |token| {
        try writer.print("{}\n", .{token});
    }
}

const ErrorReporter = struct {
    had_errors: bool = false,
    fn report(self: *ErrorReporter, line: i32, message: []const u8) void {
        std.debug.print("[line {}] Error: {s}\n", .{ line, message });
        self.had_errors = true;
    }
    fn reset(self: *ErrorReporter) void {
        self.had_errors = false;
    }
};

const Token = struct {
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
            else => self.reporter.report(self.line, "Unexpected character."),
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
            self.reporter.report(self.line, "Unterminated string.");
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
