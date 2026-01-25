"""Tests for code analyzer."""
import pytest
from tools.code_analyzer import CodeAnalyzer, CodeMetrics, LanguageConfig, LANGUAGE_CONFIGS


@pytest.fixture
def analyzer():
    return CodeAnalyzer()


# =============================================================================
# Language Config Tests
# =============================================================================

class TestLanguageConfig:
    """Tests for LanguageConfig dataclass."""

    def test_all_supported_languages_have_configs(self):
        """Verify all main languages have configs."""
        expected = ["python", "javascript", "typescript", "java", "go", "rust"]
        for lang in expected:
            assert lang in LANGUAGE_CONFIGS

    def test_config_structure(self):
        """Verify config structure."""
        for name, config in LANGUAGE_CONFIGS.items():
            assert isinstance(config, LanguageConfig)
            assert config.name == name
            assert len(config.extensions) > 0
            assert config.comment_single is not None

    def test_python_config(self):
        """Test Python config specifics."""
        config = LANGUAGE_CONFIGS["python"]
        assert ".py" in config.extensions
        assert len(config.function_patterns) > 0
        assert len(config.class_patterns) > 0
        assert len(config.issue_patterns) > 0

    def test_java_config(self):
        """Test Java config specifics."""
        config = LANGUAGE_CONFIGS["java"]
        assert ".java" in config.extensions
        assert len(config.interface_patterns) > 0
        assert len(config.enum_patterns) > 0
        assert len(config.issue_patterns) > 0

    def test_go_config(self):
        """Test Go config specifics."""
        config = LANGUAGE_CONFIGS["go"]
        assert ".go" in config.extensions
        assert len(config.interface_patterns) > 0
        assert len(config.struct_patterns) > 0
        assert len(config.issue_patterns) > 0

    def test_rust_config(self):
        """Test Rust config specifics."""
        config = LANGUAGE_CONFIGS["rust"]
        assert ".rs" in config.extensions
        assert len(config.struct_patterns) > 0
        assert len(config.enum_patterns) > 0


@pytest.fixture
def python_code():
    return '''# Sample Python code
import os
from typing import List

class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b

def main():
    calc = Calculator()
    print(calc.add(1, 2))

if __name__ == "__main__":
    main()
'''


class TestCodeAnalyzer:
    """Tests for CodeAnalyzer."""

    def test_detect_language_python(self, analyzer):
        assert analyzer.detect_language("test.py") == "python"

    def test_detect_language_javascript(self, analyzer):
        assert analyzer.detect_language("app.js") == "javascript"

    def test_detect_language_typescript(self, analyzer):
        assert analyzer.detect_language("component.tsx") == "typescript"

    def test_detect_language_unknown(self, analyzer):
        assert analyzer.detect_language("file.xyz") == "unknown"

    def test_extract_metrics_python(self, analyzer, python_code):
        metrics = analyzer.extract_metrics(python_code, "python")

        assert metrics.lines_total > 0
        assert metrics.functions >= 3  # add, multiply, main
        assert metrics.classes >= 1
        assert metrics.imports >= 1

    def test_extract_functions_python(self, analyzer, python_code):
        functions = analyzer.extract_functions(python_code, "python")

        assert len(functions) >= 3
        names = [f["name"] for f in functions]
        assert "add" in names
        assert "multiply" in names
        assert "main" in names

    def test_find_potential_issues_todo(self, analyzer):
        code = "# TODO: Fix this later\ndef foo(): pass"
        issues = analyzer.find_potential_issues(code, "python")

        assert len(issues) >= 1
        assert any("TODO" in i["message"] for i in issues)

    def test_find_potential_issues_hardcoded_password(self, analyzer):
        code = 'password = "secret123"'
        issues = analyzer.find_potential_issues(code, "python")

        assert len(issues) >= 1
        assert any("password" in i["message"].lower() for i in issues)

    def test_find_potential_issues_clean_code(self, analyzer):
        code = """
def add(a, b):
    return a + b
"""
        issues = analyzer.find_potential_issues(code, "python")
        # Should have minimal issues for clean code
        critical_issues = [i for i in issues if i["severity"] == "critical"]
        assert len(critical_issues) == 0


# =============================================================================
# Java Analysis Tests
# =============================================================================

class TestJavaAnalysis:
    """Tests for Java code analysis."""

    @pytest.fixture
    def java_code(self):
        return '''package com.example;

import java.util.List;
import java.util.ArrayList;

public interface UserService {
    User findById(Long id);
    List<User> findAll();
}

public enum Status {
    ACTIVE,
    INACTIVE,
    PENDING
}

public class UserServiceImpl implements UserService {
    private final List<User> users = new ArrayList<>();

    @Override
    public User findById(Long id) {
        return users.stream()
            .filter(u -> u.getId().equals(id))
            .findFirst()
            .orElse(null);
    }

    @Override
    public List<User> findAll() {
        return new ArrayList<>(users);
    }

    private void logAccess(String action) {
        System.out.println("Access: " + action);
    }
}
'''

    def test_extract_metrics_java(self, analyzer, java_code):
        """Test Java metrics extraction."""
        metrics = analyzer.extract_metrics(java_code, "java")

        assert metrics.lines_total > 0
        assert metrics.classes >= 1
        assert metrics.interfaces >= 1
        assert metrics.enums >= 1
        assert metrics.imports >= 2

    def test_extract_functions_java(self, analyzer, java_code):
        """Test Java function extraction."""
        functions = analyzer.extract_functions(java_code, "java")

        assert len(functions) >= 2
        names = [f["name"] for f in functions]
        assert "findById" in names or "findAll" in names

    def test_extract_interfaces_java(self, analyzer, java_code):
        """Test Java interface extraction."""
        interfaces = analyzer.extract_interfaces(java_code, "java")

        assert len(interfaces) >= 1
        names = [i["name"] for i in interfaces]
        assert "UserService" in names

    def test_find_java_issues(self, analyzer):
        """Test finding Java-specific issues."""
        code = '''
public class Example {
    public void process() {
        try {
            doSomething();
        } catch (Exception e) {
            e.printStackTrace();
        }
        Random rand = new Random();
        int value = rand.nextInt();
    }
}
'''
        issues = analyzer.find_potential_issues(code, "java")

        assert len(issues) >= 1
        messages = [i["message"].lower() for i in issues]
        assert any("exception" in m or "random" in m or "printstacktrace" in m for m in messages)


# =============================================================================
# Go Analysis Tests
# =============================================================================

class TestGoAnalysis:
    """Tests for Go code analysis."""

    @pytest.fixture
    def go_code(self):
        return '''package main

import (
    "fmt"
    "net/http"
)

type User struct {
    ID   int
    Name string
}

type UserRepository interface {
    FindByID(id int) (*User, error)
    Save(user *User) error
}

func NewUser(id int, name string) *User {
    return &User{ID: id, Name: name}
}

func (u *User) String() string {
    return fmt.Sprintf("User{ID: %d, Name: %s}", u.ID, u.Name)
}

func main() {
    user := NewUser(1, "Alice")
    fmt.Println(user.String())
}
'''

    def test_extract_metrics_go(self, analyzer, go_code):
        """Test Go metrics extraction."""
        metrics = analyzer.extract_metrics(go_code, "go")

        assert metrics.lines_total > 0
        assert metrics.functions >= 2
        assert metrics.interfaces >= 1
        assert metrics.structs >= 1
        assert metrics.imports >= 1

    def test_extract_functions_go(self, analyzer, go_code):
        """Test Go function extraction."""
        functions = analyzer.extract_functions(go_code, "go")

        assert len(functions) >= 2
        names = [f["name"] for f in functions]
        assert "NewUser" in names
        assert "main" in names

    def test_extract_interfaces_go(self, analyzer, go_code):
        """Test Go interface extraction."""
        interfaces = analyzer.extract_interfaces(go_code, "go")

        assert len(interfaces) >= 1
        names = [i["name"] for i in interfaces]
        assert "UserRepository" in names

    def test_extract_structs_go(self, analyzer, go_code):
        """Test Go struct extraction."""
        structs = analyzer.extract_structs(go_code, "go")

        assert len(structs) >= 1
        names = [s["name"] for s in structs]
        assert "User" in names

    def test_find_go_issues(self, analyzer):
        """Test finding Go-specific issues."""
        code = '''
package main

import "crypto/md5"

func process(data []byte) {
    hash := md5.Sum(data)
    fmt.Println(hash)
    panic("unexpected error")
}
'''
        issues = analyzer.find_potential_issues(code, "go")

        assert len(issues) >= 1
        messages = [i["message"].lower() for i in issues]
        assert any("md5" in m or "panic" in m or "print" in m for m in messages)


# =============================================================================
# TypeScript Analysis Tests
# =============================================================================

class TestTypeScriptAnalysis:
    """Tests for TypeScript code analysis."""

    @pytest.fixture
    def typescript_code(self):
        return '''import { Component } from '@angular/core';

interface UserDTO {
    id: number;
    name: string;
    email?: string;
}

enum Role {
    Admin = 'ADMIN',
    User = 'USER',
    Guest = 'GUEST'
}

class UserService {
    private users: UserDTO[] = [];

    findById(id: number): UserDTO | undefined {
        return this.users.find(u => u.id === id);
    }

    create(user: UserDTO): void {
        this.users.push(user);
    }
}

const formatUser = (user: UserDTO): string => {
    return `${user.name} (${user.id})`;
};
'''

    def test_extract_metrics_typescript(self, analyzer, typescript_code):
        """Test TypeScript metrics extraction."""
        metrics = analyzer.extract_metrics(typescript_code, "typescript")

        assert metrics.lines_total > 0
        assert metrics.classes >= 1
        assert metrics.interfaces >= 1
        assert metrics.enums >= 1
        assert metrics.imports >= 1

    def test_extract_interfaces_typescript(self, analyzer, typescript_code):
        """Test TypeScript interface extraction."""
        interfaces = analyzer.extract_interfaces(typescript_code, "typescript")

        assert len(interfaces) >= 1
        names = [i["name"] for i in interfaces]
        assert "UserDTO" in names

    def test_find_typescript_issues(self, analyzer):
        """Test finding TypeScript-specific issues."""
        code = '''
function process(data: any): any {
    // @ts-ignore
    console.log(data);
    return data;
}
'''
        issues = analyzer.find_potential_issues(code, "typescript")

        assert len(issues) >= 1
        messages = [i["message"].lower() for i in issues]
        assert any("any" in m or "ts-ignore" in m or "console" in m for m in messages)


# =============================================================================
# Rust Analysis Tests
# =============================================================================

class TestRustAnalysis:
    """Tests for Rust code analysis."""

    @pytest.fixture
    def rust_code(self):
        return '''use std::collections::HashMap;

struct Config {
    name: String,
    value: i32,
}

enum Status {
    Active,
    Inactive,
    Unknown,
}

fn create_config(name: &str, value: i32) -> Config {
    Config {
        name: name.to_string(),
        value,
    }
}

fn get_status(code: i32) -> Status {
    match code {
        1 => Status::Active,
        2 => Status::Inactive,
        _ => Status::Unknown,
    }
}

fn main() {
    let config = create_config("test", 42);
    println!("Config: {}", config.name);
}
'''

    def test_extract_metrics_rust(self, analyzer, rust_code):
        """Test Rust metrics extraction."""
        metrics = analyzer.extract_metrics(rust_code, "rust")

        assert metrics.lines_total > 0
        assert metrics.functions >= 2
        assert metrics.structs >= 1
        assert metrics.enums >= 1
        assert metrics.imports >= 1

    def test_extract_structs_rust(self, analyzer, rust_code):
        """Test Rust struct extraction."""
        structs = analyzer.extract_structs(rust_code, "rust")

        assert len(structs) >= 1
        names = [s["name"] for s in structs]
        assert "Config" in names

    def test_find_rust_issues(self, analyzer):
        """Test finding Rust-specific issues."""
        code = '''
fn main() {
    let result = some_operation().unwrap();
    println!("Result: {}", result);
    unsafe {
        do_something_dangerous();
    }
}
'''
        issues = analyzer.find_potential_issues(code, "rust")

        assert len(issues) >= 1
        messages = [i["message"].lower() for i in issues]
        assert any("unwrap" in m or "unsafe" in m or "println" in m for m in messages)


# =============================================================================
# Full Analysis Tests
# =============================================================================

class TestGetFullAnalysis:
    """Tests for get_full_analysis method."""

    def test_full_analysis_python(self, analyzer, python_code):
        """Test full analysis for Python."""
        result = analyzer.get_full_analysis(python_code, "python")

        assert "language" in result
        assert result["language"] == "python"
        assert "metrics" in result
        assert "functions" in result
        assert "interfaces" in result
        assert "structs" in result
        assert "issues" in result
        assert "issue_summary" in result

    def test_full_analysis_metrics_structure(self, analyzer, python_code):
        """Test metrics structure in full analysis."""
        result = analyzer.get_full_analysis(python_code, "python")
        metrics = result["metrics"]

        expected_keys = [
            "lines_total", "lines_code", "lines_comment", "lines_blank",
            "functions", "classes", "interfaces", "structs", "enums", "imports"
        ]
        for key in expected_keys:
            assert key in metrics

    def test_full_analysis_issue_summary(self, analyzer):
        """Test issue summary in full analysis."""
        code = '''
password = "secret123"
# TODO: refactor this
eval(user_input)
'''
        result = analyzer.get_full_analysis(code, "python")
        summary = result["issue_summary"]

        assert "critical" in summary
        assert "warning" in summary
        assert "info" in summary
        assert "total" in summary
        assert summary["total"] == len(result["issues"])

    def test_full_analysis_java(self, analyzer):
        """Test full analysis for Java."""
        java_code = '''
public interface Service {
    void process();
}

public enum Type { A, B }

public class Impl implements Service {
    public void process() {
        System.out.println("processing");
    }
}
'''
        result = analyzer.get_full_analysis(java_code, "java")

        assert result["language"] == "java"
        assert result["metrics"]["interfaces"] >= 1
        assert result["metrics"]["enums"] >= 1
        assert len(result["interfaces"]) >= 1

    def test_full_analysis_go(self, analyzer):
        """Test full analysis for Go."""
        go_code = '''
package main

type Handler interface {
    Handle() error
}

type Request struct {
    ID string
}

func NewRequest(id string) *Request {
    return &Request{ID: id}
}
'''
        result = analyzer.get_full_analysis(go_code, "go")

        assert result["language"] == "go"
        assert result["metrics"]["interfaces"] >= 1
        assert result["metrics"]["structs"] >= 1
        assert len(result["interfaces"]) >= 1
        assert len(result["structs"]) >= 1
