# dc_simulation

####
To run the code, you need python3.x and a multi-agent simulation platform named Mesa. Type: python src/model.py

Notice: 
	1. Since the experimental treatments change and it is hard to know what the treatments are in advance, so when you have 
	   new treatment you may need to modify getBatchConfig, getNetworkConfig, and expSummary in src/utils.py

	2. I assume there are 20 regular players (include visible color players). If not, you may need to modify expSummary in src/utils.py 
