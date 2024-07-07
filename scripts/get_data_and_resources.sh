git clone https://github.com/ryokamoi/wice.git ../wice
git clone https://github.com/sheffieldnlp/fever-scorer ../fever_scorer
cd ..
mkdir fever
cd fever
wget https://fever.ai/download/fever/paper_test.jsonl
wget -O sampled_ids.json https://raw.githubusercontent.com/RuochenZhao/Verify-and-Edit/main/Fever/data/sampled_ids.json
wget https://fever.ai/download/fever/wiki-pages.zip
unzip wiki-pages.zip