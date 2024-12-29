# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

# Project structure
SRCDIR = .
BUILDDIR = build
TARGET = $(BUILDDIR)/cnn

# Source files
SRCFILES = main.cpp layers/conv_layer.cpp layers/relu_layer.cpp utils/tensor.cpp utils/mnist_loader.cpp layers/fully_connected_layer.cpp layers/flatten_layer.cpp layers/maxpool_layer.cpp utils/loss.cpp utils/optimizer.cpp layers/relu_activation.cpp
OBJFILES = $(patsubst %.cpp, $(BUILDDIR)/%.o, $(SRCFILES))

# Default rule
all: $(TARGET)

# Linking
$(TARGET): $(OBJFILES)
	@if not exist "$(BUILDDIR)" mkdir $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compilation
$(BUILDDIR)/%.o: %.cpp
	@for %%d in ("$(@D)") do if not exist %%d mkdir %%d
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	@if exist "$(BUILDDIR)" rmdir /S /Q $(BUILDDIR)

# Run the program
run: $(TARGET)
	$(TARGET)
