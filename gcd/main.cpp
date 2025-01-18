#include <iostream>

int gcd(int a, int b) {                                                                                                    
    if (a < b) {                                                                                                           
        std::swap(a, b);
    }

    while (b > 0) {                                                                                                        
        a %= b;                                                                                                            
        std::swap(a, b);
    }

    return a;                                                                                                              
}



int main(int argc, char** argv)                                                                                            
{
    int a = 30, b = 130;                                                                                                   

    std::cout << gcd(a, b) << std::endl;                                                                                   

    return 0;                                                                                                              
}

