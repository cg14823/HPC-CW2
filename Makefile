# Makefile

EXE=d2q9-bgk

CC=mpicc
CFLAGS= -std=c99 -O3 -ffast-math -ftree-vectorizer-verbose
LIBS = -lm

SIZE =128x128

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/$(SIZE).final_state.dat
REF_AV_VELS_FILE=check/$(SIZE).av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ -o $@

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)
