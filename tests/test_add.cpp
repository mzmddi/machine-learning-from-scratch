

#include <catch2/catch_test_macros.hpp>

int add(int a, int b)
{
    return a + b;
}

TEST_CASE("Basic math operations work", "[math]")
{
    REQUIRE(add(2, 2) == 4);
    REQUIRE(add(-1, 1) == 0);
}

TEST_CASE("Intentional assertion check", "[sanity]")
{
    SECTION("Checking boolean truth")
    {
        bool is_working = true;
        CHECK(is_working == true);
    }

    SECTION("Checking integer equality")
    {
        int expected = 42;
        REQUIRE(expected == 42);
    }
}