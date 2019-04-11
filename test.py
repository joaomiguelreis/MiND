import numpy as np
from sklearn import svm


v = []
u = []

for i in range(10):
	v.append(np.random.rand(5))
	u.append(np.random.rand(5)*np.random.normal(0,1,5))
v_label = np.zeros(len(v))
u_label = np.ones(len(u))

labels = np.concatenate([u_label, v_label])
u = np.concatenate([u,v])

clf = svm.SVC()
clf.fit(u,labels)

print(clf.predict([np.random.rand(5)]))