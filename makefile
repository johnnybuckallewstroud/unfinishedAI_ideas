#How to use it:
#To Compile: 
#Open your terminal in the project folder and type:
#Bash
#make
#This will generate the evialin executable using the specified gcc flags.
#To Install: 
#Run the following command (usually requires sudo for system directories):
#Bash
#sudo make install
#This copies the program to /usr/local/bin, allowing you to run evialin from any terminal directory.
#To Uninstall:
#Bash
#sudo make uninstall
#Important Dependency Note,
#The source code indicates that this program requires several external libraries. 
#Before running make, ensure you have the development headers installed:
#
#GTK+ 3.0: libgtk-3-dev
#SQLite3: libsqlite3-dev
#Math and Pthread libraries: 
#These are typically included by default on most Linux systems.

# Compiler and project name
CC = gcc
TARGET = evialin
SRC = evialin_gui.c

# Compilation flags derived from the source header
# pkg-config automatically handles GTK+ 3.0 headers and libraries
CFLAGS = `pkg-config --cflags gtk+-3.0` -Wall
LIBS = `pkg-config --libs gtk+-3.0` -lm -lpthread -lsqlite3

# Installation paths
PREFIX = /usr/local
BINDIR = $(PREFIX)/bin

# Dev mode prototype building of project
dev: $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS) $(LIBS)
	./$(TARGET)

run: $(TARGET)
	./$(TARGET)
# Default build rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS) $(LIBS)

# Installation rule
install: $(TARGET)
	@echo "Installing $(TARGET) to $(BINDIR)..."
	install -d $(BINDIR)
	install -m 755 $(TARGET) $(BINDIR)
	@echo "Installation complete."

# Uninstallation rule
uninstall:
	@echo "Removing $(TARGET) from $(BINDIR)..."
	rm -f $(BINDIR)/$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

.PHONY: all dev run install uninstall clean
