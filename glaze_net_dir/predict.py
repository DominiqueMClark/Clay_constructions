from tensorflow.keras.models import load_model
import numpy as np 
glazey = "E: 60 lithium carbonate E: 40 talc E: 6 manganese dioxide E: 2 Bentonite E: 2 frit,Ferro 3124"

features = ['chrome', 'nickel', 'quartz', 'alumina', 'zirconia', 'kaolin,epk', 'feldspar,f-4', 'ash,hardwood', 'feldspar,potash', 'nelson', 'ash,bone', 'feldspar,forshammer', 'epsom', 'cobalt', 'nepheline', 'frit,johnson', 'iron', 'crocus', 'titanium', 'vitrox', 'talc', 'frit,pemco', 'silica', 'feldspar,nc-4', 'clay,barnard', 'lithium', 'frit,ferro', 'frit,hommel', 'clay,oldenwalder', 'selsor', 'kaolin', 'wine', 'silicon', 'feldspar,feldshammer', 'nceca', 'feldspar,a-3', 'frit,fusion', 'stain', 'manganese', 'edwards', 'slip,sheffield', 'kendall', 'ochre', 'redart', 'kaolin,grolleg', 'slip,albany', 'wollastonite', 'borax', 'clay,vitrox', 'rutile', 'zinc', 'colemanite', 'ball', 'feldspar,k-200', 'chappel', 'dolomite', 'spodumene,low-melt', 'copper', 'feldspar,c-6', 'hansen', 'calcium', 'borate,gerstley', 'gum,cmc', 'burkett/borcherding', 'strontium', 'clarke', 'tin', 'stone,cornwall', 'calcite', 'spodumene', 'cryolite', 'ilmenite', 'slip,barnard', 'feldspar,kingman', 'ash,soda', 'kaolin,calcined', 'barium', 'kaolin,mcnamee', 'sodium', 'pike', 'feldspar,g-200', 'behrens', 'feldspar,lepidolite', 'frit,potterycrafts', 'feldspar,soda', 'bentonite', 'magnesium', 'feldspar,custer', 'clay,redart']
labels = ['pitted', 'cratered', 'rough', 'metallic', 'underfired', 'matte', 'gloss', 'crystalline', 'semigloss', 'waxy', 'semimatte', 'satin']
glaze_network = load_model("glaze_network.h5")

def predict_glaze(glaze):
	lis = np.zeros((1, len(features)))

	x = glaze.split("E:")
	x.remove('')
	num = dict(list(enumerate(features)))
	num = dict([(value,key) for key, value in num.items()])
	for material in x:
		y = material.split(' ')
		if y[2].lower() in num:
			lis[0][num[y[2].lower()]] = y[1]
		else:
			print("incorrect material")

	pred = np.argmax(glaze_network.predict(lis))
	return (labels[pred])
'''pred = predict_glaze(glazey)
print(pred)'''
