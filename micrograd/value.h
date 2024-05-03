#ifndef VALUE_H

#include <cmath>
#include <functional>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

template <typename T>
class Value {
 public:
  Value(
      T data, std::string op, std::vector<Value<T>*> children,
      std::string name = "")
      : data_(data), grad_(0), op_(op), children_(children), name_(name) {}

  Value(T data, std::string name) : data_(data), grad_(0), op_(""), name_(name) {}
  Value(T data) : data_(data), grad_(0), op_(""), name_("") {}

  friend std::ostream& operator<<(std::ostream& os, const Value& val) {
    os << val.name_ << " = " << val.data_ << ", grad = " << val.grad_
       << ", op = " << val.op_ << " " << (val.backwards_ != nullptr) << std::endl;
    return os;
  }

  void SetName(std::string name) { name_ = name; }

  Value operator+(Value& other) {
    std::vector<Value<T>*> children = {this, &other};
    auto parent = Value(data_ + other.data_, "+", children);

    parent.backwards_ = [&parent, &other, this]() {
      this->grad_ += parent.grad_;
      other.grad_ += parent.grad_;
    };

    return parent;
  }

  Value operator+(T scalar) {
    return *this + Value(scalar);
  }

  Value operator*(Value& other) {
    std::vector<Value<T>*> children = {this, &other};
    auto parent = Value(data_ * other.data_, "*", children);
    parent.backwards_ = [&parent, &other, this]() {
      this->grad_ += parent.grad_ * other.data_;
      other.grad_ += parent.grad_ * this->data_;
    };

    return parent;
  }

  Value operator*(T scalar) {
    return *this * Value(scalar);
  }

  Value operator-(Value& other) {
    return *this + (other * -1);
  }

  Value operator-(T scalar) {
    return *this - Value(scalar);
  }

  Value operator^(T exp) {
    std::vector<Value<T>*> children = {this};
    auto parent = Value(std::pow(data_, exp), "^", children);
    parent.backwards_ = [&parent, exp, this]() {
      this->grad_ += parent.grad_ * exp * std::pow(this->data_, exp - 1);
    };

    return parent;
  }

  Value operator/(Value& other) {
    return *this * (other ^ -1);
  }

  Value Tanh() {
    auto parent = Value(std::tanh(data_), "tanh", {this});
    parent.backwards_ = [&parent, this]() {
      this->grad_ += parent.grad_ * (1 - std::pow(parent.data_, 2));
    };
    return parent;
  }

  Value ReLU() {
    auto parent = Value(data_ > 0 ? data_ : 0, "ReLU", {this});
    parent.backwards_ = [&parent, this]() {
      this->grad_ += data_ > 0 ? parent.grad_ : 0;
    };
    return parent;
  }

  Value Sigmoid() {
    auto parent = std::exp(data_) / (1 + std::exp(data_));
    parent.backwards_ = [&parent, this]() {
      this->grad_ += parent.grad_ * parent.data_ * (1 - parent.data_);
    };
  }

  void Backward() {
    // Topological sort.
    std::vector<const Value<T>*> out;
    std::unordered_set<const Value<T>*> visited;
    std::queue<const Value<T>*> q;
    q.push(this);

    while (!q.empty()) {
      auto node = q.front();
      q.pop();

      // Already visited. Skip.
      if (visited.find(node) != visited.end()) {
        continue;
      }

      out.push_back(node);
      visited.insert(node);
      for (auto child : node->GetChildren()) {
        q.push(child);
      }
    }

    grad_ = 1;
    for (auto node : out) {
      if (node->backwards_ != nullptr) {
        node->backwards_();
      }
      std::cout << *node << std::endl;
    }
  }

  std::vector<Value<T>*> GetChildren() const { return children_; }

 private:
  T data_;
  T grad_;
  std::string op_;
  std::vector<Value<T>*> children_;
  std::function<void()> backwards_;
  std::string name_;
};

template <typename T>
void Visualize(const Value<T>& val) {
  std::cout << val;
  for (auto child : val.GetChildren()) {
    std::cout << std::endl;
    Visualize(*child);
  }
}

#endif  // VALUE_H
