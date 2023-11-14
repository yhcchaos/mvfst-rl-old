#include <iostream>

class MyClass {
public:
    static int staticVar; // 静态成员变量的声明

    MyClass(int value) {
        staticVar = value; // 在构造函数中初始化静态成员变量
    }
};

// 初始化静态成员变量
//int MyClass::staticVar = 0;

// 外部函数，用于访问静态成员变量
void externalFunction() {
    std::cout << "Static Variable: " << MyClass::staticVar << std::endl;
}

int main() {
    MyClass obj(42);
    externalFunction(); // 访问静态成员变量

    return 0;
}