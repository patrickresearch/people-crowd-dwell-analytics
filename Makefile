SEQ ?= MOT17-04
MODEL ?= yolo26s.pt
IMGSZ ?= 1280
SRC := data/MOT17/train/$(SEQ)-FRCNN
RUN := runs/$(SEQ)-$(basename $(MODEL))-$(IMGSZ)

.PHONY: setup data preview track analyze eval eval-all bench gif test lint

setup:
	uv sync --all-extras

data:
	mkdir -p data
	curl -L -C - -o data/MOT17.zip https://motchallenge.net/data/MOT17.zip
	unzip -qn data/MOT17.zip 'MOT17/train/*-FRCNN/*' -d data

preview:
	uv run crowd-zones $(SRC) --scene $(SEQ) --out runs/$(SEQ)-preview.jpg

track:
	uv run crowd-track $(SRC) --scene $(SEQ) --model $(MODEL) --imgsz $(IMGSZ) --out $(RUN)

analyze:
	uv run crowd-analyze $(RUN)

eval:
	uv run crowd-eval $(SRC)/gt/gt.txt $(RUN)/tracks.csv $(RUN)/tracks_stitched.csv

eval-all:
	uv run crowd-eval $(SRC)/gt/gt.txt runs/$(SEQ)-*/tracks*.csv

bench:
	uv run crowd-bench $(SRC) --imgsz $(IMGSZ)

gif:
	ffmpeg -y -loglevel error -ss 0 -t 6 -i $(RUN)/annotated.mp4 \
		-vf "fps=10,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
		assets/demo.gif
	cp $(RUN)/heatmap.png $(RUN)/dwell_hist.png $(RUN)/throughput.png assets/
	du -h assets/*

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .
