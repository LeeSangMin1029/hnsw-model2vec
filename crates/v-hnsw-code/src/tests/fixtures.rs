//! Shared sample code fixtures for multi-language chunker tests.

pub const SAMPLE_RUST: &str = r#"
use std::collections::HashMap;
use crate::types::Payment;

/// Process a payment through the gateway.
///
/// Validates amount and creates a payment intent.
pub fn process_payment(amount: f64, currency: &str) -> Result<PaymentIntent, Error> {
    validate_amount(amount)?;
    let intent = stripe::create_intent(amount, currency)?;
    db::insert(&intent)?;
    Ok(intent)
}

/// Payment data structure.
#[derive(Debug, Clone)]
pub struct PaymentIntent {
    pub id: String,
    pub amount: f64,
    pub currency: String,
}

impl PaymentIntent {
    /// Create a new payment intent.
    pub fn new(id: String, amount: f64, currency: String) -> Self {
        Self { id, amount, currency }
    }

    /// Check if the payment is valid.
    fn is_valid(&self) -> bool {
        self.amount > 0.0
    }
}

enum PaymentStatus {
    Pending,
    Completed,
    Failed,
}
"#;

pub const SAMPLE_RUST_MINI: &str = r#"
pub fn hello() -> String {
    String::from("hi")
}
"#;

pub const SAMPLE_TS: &str = r#"
import { Request, Response } from 'express';
import { UserService } from './services';

/** Process an incoming HTTP request. */
export function handleRequest(req: Request, res: Response): Promise<void> {
    const user = UserService.findById(req.params.id);
    validate(user);
    res.json(user);
}

/** A user data transfer object. */
export interface UserDTO {
    id: string;
    name: string;
    email: string;
}

export class UserController {
    private service: UserService;

    constructor(service: UserService) {
        this.service = service;
    }

    /** Get a user by ID. */
    async getUser(id: string): Promise<UserDTO> {
        return this.service.findById(id);
    }

    /** Delete a user. */
    async deleteUser(id: string): Promise<void> {
        await this.service.delete(id);
    }
}

export enum Role {
    Admin = 'admin',
    User = 'user',
    Guest = 'guest',
}
"#;

pub const SAMPLE_PY: &str = r#"
import os
from typing import Optional, List
from dataclasses import dataclass

def process_data(items: List[str], limit: int = 10) -> Optional[dict]:
    """Process a list of data items.

    Args:
        items: The input items to process.
        limit: Maximum number of items.

    Returns:
        A dictionary of processed results.
    """
    validated = validate_items(items)
    return aggregate(validated, limit)

@dataclass
class DataResult:
    """Result of data processing."""
    count: int
    items: List[str]
    status: str = "pending"

    def is_complete(self) -> bool:
        """Check if processing is complete."""
        return self.status == "complete"

    def summary(self) -> str:
        """Get a summary string."""
        return f"{self.count} items: {self.status}"

class DataProcessor:
    """Processes data with configuration."""

    def __init__(self, config: dict):
        self.config = config

    def run(self, data: List[str]) -> DataResult:
        """Execute the processing pipeline."""
        result = process_data(data, self.config.get("limit", 10))
        return DataResult(count=len(data), items=data)

def _private_helper(x: int) -> int:
    return x * 2
"#;

pub const SAMPLE_GO: &str = r#"
package main

import (
    "fmt"
    "net/http"
)

// HandleRequest processes an incoming HTTP request.
func HandleRequest(w http.ResponseWriter, r *http.Request) error {
    data := ParseBody(r)
    if err := Validate(data); err != nil {
        return err
    }
    fmt.Fprintf(w, "OK")
    return nil
}

// RequestData holds parsed request information.
type RequestData struct {
    Method  string
    Path    string
    Headers map[string]string
}

// String returns a string representation.
func (rd *RequestData) String() string {
    return fmt.Sprintf("%s %s", rd.Method, rd.Path)
}

// Validate checks if the data is valid.
func (rd *RequestData) Validate() error {
    if rd.Method == "" {
        return fmt.Errorf("empty method")
    }
    return nil
}

// Handler is an interface for request handlers.
type Handler interface {
    ServeHTTP(w http.ResponseWriter, r *http.Request)
    Name() string
}

// Status represents processing status.
type Status int

const (
    StatusPending Status = iota
    StatusDone
    StatusFailed
)
"#;

pub const SAMPLE_JAVA: &str = r#"
package com.example.service;

import java.util.List;
import java.util.Optional;

/**
 * Processes payment transactions.
 */
public class PaymentService {

    private final PaymentGateway gateway;

    public PaymentService(PaymentGateway gateway) {
        this.gateway = gateway;
    }

    /**
     * Process a payment with the given amount.
     */
    public PaymentResult processPayment(double amount, String currency) {
        validate(amount);
        PaymentIntent intent = gateway.createIntent(amount, currency);
        return new PaymentResult(intent.getId(), "success");
    }

    private void validate(double amount) {
        if (amount <= 0) {
            throw new IllegalArgumentException("Invalid amount");
        }
    }
}

/**
 * Result of a payment operation.
 */
public class PaymentResult {
    private final String id;
    private final String status;

    public PaymentResult(String id, String status) {
        this.id = id;
        this.status = status;
    }

    public String getId() {
        return id;
    }
}

public interface PaymentGateway {
    PaymentIntent createIntent(double amount, String currency);
    void cancelIntent(String intentId);
}

public enum PaymentStatus {
    PENDING,
    COMPLETED,
    FAILED,
    REFUNDED
}
"#;

pub const SAMPLE_C: &str = r#"
#include <stdio.h>
#include <stdlib.h>

/**
 * Process a data buffer.
 */
int process_buffer(const char *data, size_t len) {
    if (data == NULL) return -1;
    validate_input(data, len);
    return transform(data, len);
}

typedef struct {
    int x;
    int y;
    char name[64];
} Point;

enum Color {
    RED,
    GREEN,
    BLUE
};

static void helper_func(int n) {
    printf("helper: %d\n", n);
}
"#;

// ── Edge case fixtures ─────────────────────────────────────────────

pub const EMPTY_SOURCE: &str = "";

pub const WHITESPACE_ONLY: &str = "   \n\n   \t\t\n";

pub const SAMPLE_RUST_NESTED: &str = r#"
pub mod outer {
    pub fn outer_fn() -> i32 {
        fn inner_fn() -> i32 {
            42
        }
        inner_fn()
    }
}
"#;

pub const SAMPLE_RUST_DEEPLY_NESTED_IMPL: &str = r#"
pub struct Outer;

impl Outer {
    pub fn method_a(&self) -> i32 {
        let closure = |x: i32| {
            x + 1
        };
        closure(1)
    }

    pub fn method_b(&self, val: i32) -> String {
        format!("val={val}")
    }
}

impl std::fmt::Display for Outer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Outer")
    }
}
"#;

pub const SAMPLE_RUST_SINGLE_LINE_FN: &str = r#"
fn one_liner() { 42 }
"#;

pub const SAMPLE_PY_NESTED_CLASSES: &str = r#"
class Outer:
    """An outer class."""

    class Inner:
        """An inner class."""

        def inner_method(self) -> str:
            """Do inner things."""
            return "inner"

    def outer_method(self) -> str:
        """Do outer things."""
        return "outer"
"#;

pub const SAMPLE_PY_EMPTY_BODY: &str = r#"
def stub_function():
    pass
"#;

pub const SAMPLE_PY_SYNTAX_ERROR: &str = r#"
def broken(
    # missing closing paren
    pass
"#;

pub const SAMPLE_TS_EMPTY_CLASS: &str = r#"
export class EmptyClass {}
"#;

pub const SAMPLE_TS_ARROW_FUNCTIONS: &str = r#"
export const add = (a: number, b: number): number => a + b;

export const greet = (name: string): string => {
    return `Hello, ${name}!`;
};
"#;

pub const SAMPLE_GO_EMPTY_STRUCT: &str = r#"
package main

type EmptyStruct struct{}

func NewEmpty() EmptyStruct {
    return EmptyStruct{}
}
"#;

pub const SAMPLE_C_FORWARD_DECL: &str = r#"
struct ForwardDeclared;

int use_forward(struct ForwardDeclared *ptr) {
    return ptr != 0;
}
"#;

pub const SAMPLE_JAVA_ABSTRACT: &str = r#"
public abstract class AbstractProcessor {
    public abstract void process(String input);

    public void log(String msg) {
        System.out.println(msg);
    }
}
"#;

pub const SAMPLE_CPP: &str = r#"
#include <string>
#include <vector>
#include <memory>

/**
 * Represents a graph node.
 */
class GraphNode {
public:
    GraphNode(int id, std::string label)
        : id_(id), label_(std::move(label)) {}

    /** Get the node ID. */
    int getId() const { return id_; }

    /** Get the label. */
    std::string getLabel() const { return label_; }

    /** Add a neighbor node. */
    void addNeighbor(std::shared_ptr<GraphNode> node) {
        neighbors_.push_back(node);
        updateIndex();
    }

private:
    int id_;
    std::string label_;
    std::vector<std::shared_ptr<GraphNode>> neighbors_;

    void updateIndex() {
        // internal bookkeeping
    }
};

/** Build a graph from raw data. */
std::vector<GraphNode> buildGraph(const std::vector<int>& ids) {
    std::vector<GraphNode> nodes;
    for (auto id : ids) {
        nodes.emplace_back(id, std::to_string(id));
    }
    return nodes;
}

enum class NodeType {
    Source,
    Sink,
    Intermediate
};

struct EdgeWeight {
    double weight;
    bool directed;
};
"#;
