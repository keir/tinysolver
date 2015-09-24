tinysolver_test: tinysolver.h tinysolver_test.cc
	g++ -Wextra -o tinysolver_test tinysolver_test.cc \
	  -I../../Downloads/eigen-eigen-bdd17ee3b1b3 \
          -I../googletest/googletest/include \
          -L../googletest/bin/ \
          -lgtest_main \
          -lgtest 
