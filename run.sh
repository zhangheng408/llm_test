mkdir -p result profile

cd result
python3 -u ../sd_test.py 2>&1 | tee sd.torch2.1.dev0810
python3 -u ../llm_test.py 2>&1 | tee llm.torch2.1.dev0810
cd -

cd profile
rocprof -d . -o sd_2_1.csv --hip-trace --roctx-trace python3 sd_2_1.py
rocprof -d . -o sd_2_1_base.csv --hip-trace --roctx-trace python3 sd_2_1_base.py
rocprof -d . -o sd_1_5.csv --hip-trace --roctx-trace python3 sd_1_5.py
rocprof -d . -o llm.bloom.128.csv --hip-trace --roctx-trace python3 llm_bloom_128.py
rocprof -d . -o llm.bloom.512.csv --hip-trace --roctx-trace python3 llm_bloom_512.py
rocprof -d . -o llm.opt.128.csv --hip-trace --roctx-trace python3 llm_opt_128.py
rocprof -d . -o llm.opt.512.csv --hip-trace --roctx-trace python3 llm_opt_512.py
