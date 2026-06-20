#ifndef MATRIX_H
#define MATRIX_H

// --- NOTES ---

// --- INCLUDES ---

#include <vector>
#include <string>
#include <iostream>

// --- CODE ----

namespace mymath
{

    class Matrix
    {
        std::string name;
        int r;
        int c;

        std::vector<float> v;

    public:
        Matrix(int target_r, int target_c);
        Matrix(int target_r, int target_c, const std::vector<float> &values);
        static Matrix Zero(int target_r, int target_c);
        static Matrix Identity(int target_p);

        // --- Setters ----

        void set_name(std::string s) { this->name = s; };
        void set_value(int position, float value) { this->v[position] = value; };
        void change_r(int s) { this->r = s; };
        void change_c(int s) { this->c = s; };

        // --- Getters ---

        std::string get_name() const { return this->name; };
        std::vector<int> get_shape() const { return std::vector<int>{r, c}; };
        float get_value(int position) const { return v[position]; };

        // --- class functions ---

        void print();

        Matrix t();
        // transpose
    };

    Matrix operator+(const Matrix &one, const Matrix &two);

    template <typename T>
    Matrix operator*(const Matrix &one, T scalar);

    Matrix operator-(const Matrix &one, const Matrix &two);

    Matrix operator%(const Matrix &one, const Matrix &two);

}

#endif