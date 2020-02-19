CC = g++
CFLAGS = -g -Wextra -Wno-unused-parameter -Wno-switch -Wno-attributes -Wall -Iinc -pthread

OBJDIR = obj
INCDIR = inc
SRCDIR = src
BINDIR = bin

HFILES=$(wildcard $(INCDIR)/*.hpp)
OFILES=$(subst .cpp,.o,$(subst $(SRCDIR),$(OBJDIR),$(wildcard $(SRCDIR)/*.cpp)))

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(HFILES)
	mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

$(BINDIR)/simulator: $(OFILES)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $^ -o $@

all: $(BINDIR)/simulator

run:
	$(BINDIR)/simulator

.PHONEY=all clean simulate visualise
.DEFAULT_GOAL=all
