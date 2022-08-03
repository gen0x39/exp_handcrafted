#!/bin/bash -eu

# 取りうるパラメタの範囲
CELL_Values="0 1 2 3"

# for で総当り
for c1 in ${CELL_Values} ; do
	for c2 in ${CELL_Values} ; do
		for c3 in ${CELL_Values} ; do
			if [ -d ./arch-${c1}${c2}${c3} ]; then
				echo -n "arch-${c1}${c2}${c3}: "
				cat arch-${c1}${c2}${c3}/log.txt | grep valid_acc | head -n 200 | grep "epoch\s199"
			fi
		done
	done
done
