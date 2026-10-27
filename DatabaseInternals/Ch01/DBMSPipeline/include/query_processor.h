#pragma once
#include "interfaces.h"
#include <sstream>
#include <stdexcept>

class SimpleQueryProcessor : public IQueryProcessor {
public:
    std::unique_ptr<ExecutionPlan> parse(const std::string& raw_query) override {
        std::cout << "[PARSER] Parsing: \"" << raw_query << "\"\n";
        
        std::istringstream iss(raw_query);
        std::string op;
        iss >> op;

        ExecutionPlan plan;
        if (op == "READ") {
            plan.op = OpType::READ;
            iss >> plan.target_id;
        } else if (op == "WRITE") {
            plan.op = OpType::WRITE;
            iss >> plan.target_id >> plan.payload;
        } else {
            throw std::invalid_argument("Unknown operation: " + op);
        }

        std::cout << "[PARSER] Generated ExecutionPlan: " << op << " " << plan.target_id << "\n";
        return std::make_unique<ExecutionPlan>(plan);
    }
};
