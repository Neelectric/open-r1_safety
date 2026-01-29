


MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_sciencev00.01
VERSION=v00.01

bash gpqa_diamond.sh $MODEL v00.01-step-000002732
bash gpqa_diamond.sh $MODEL v00.01-step-000005464
bash gpqa_diamond.sh $MODEL v00.01-step-000008196
bash gpqa_diamond.sh $MODEL v00.01-step-000010928
bash gpqa_diamond.sh $MODEL v00.01-step-000013660
bash gpqa_diamond.sh $MODEL v00.01-step-000016392
bash gpqa_diamond.sh $MODEL v00.01-step-000019124
bash gpqa_diamond.sh $MODEL v00.01-step-000021856
bash gpqa_diamond.sh $MODEL v00.01-step-000024588
bash gpqa_diamond.sh $MODEL v00.01-step-000027318
bash gpqa_diamond.sh $MODEL main

