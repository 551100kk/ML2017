import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def modify(path_mean, path_ans):
	best = pd.read_csv(path_mean)
	large = best[best.price_doc > 20000000].index
	best.loc[large, 'price_doc'] *=	1.1

	result = pd.DataFrame({'id': best.id, 'price_doc': best.price_doc})
	result.to_csv(path_ans, index=False)