#include "DecisionTreeClassifier.h"
#define inf 1e9


void DecisionTreeClassifier::treeRec(tree * t)
{
	nodes++;
	double maPers = 0.5;
	double Cp = double(t->fq) / (t->fq + t->mq);
	if (double(t->fq) / (t->fq + t->mq)>=leafPers) {
		t->claSs = 0;
		return;
	}
	if (double(t->mq) / (t->fq + t->mq)>=leafPers) {
		t->claSs = 1;
		return;
	}
	if (t->depth == maxDepth) {
		t->claSs = t->mq > t->fq;
		return;
	}
	if (nodes >= maxNodes) {
		t->claSs = t->mq > t->fq;
		return;
	}
	int splitInd; double splitEdge;
	vector< vector< int > > maq = vector< vector< int > >(2, vector<int>(2, 0));
	vector<int> cu = vector<int>(u.size(), 0);
	getSplit(splitInd, splitEdge, maPers);
	t->splitEdge = splitEdge, t->splitInd = splitInd;
	double prma = prMas[splitInd], prmi = prMis[splitInd];
	prMas[splitInd] = splitEdge;
	for (int i = 0; i < data.size(); ++i) {
		if (u[i]) continue;
		double w = data[i][splitInd];
		if (w > splitEdge)
			cu[i] = 1, u[i] = 1;
		maq[target[i]][w > splitEdge]++;
	}
	t->l = new tree(maq[0][0], maq[1][0], t->depth + 1);
	treeRec(t->l);
	prMas[splitInd] = prma;
	for (int i = 0; i < data.size(); ++i) {
		if (cu[i])
			u[i] = 0, cu[i] = 0;
	}
	prMis[splitInd] = splitEdge;
	t->r = new tree(maq[0][1], maq[1][1], t->depth + 1);
	for (int i = 0; i < data.size(); ++i) {
		if (u[i]) continue;
		double w = data[i][splitInd];
		if (w <= splitEdge)
			cu[i] = 1, u[i] = 1;
	}
	treeRec(t->r);
	for (int i = 0; i < data.size(); ++i) {
		if (cu[i])
			u[i] = 0, cu[i] = 0;
	}
	prMis[splitInd] = prmi;
}

int DecisionTreeClassifier::prRec(tree *t)
{
	if (t->claSs != -1) return t->claSs;
	if (prData[t->splitInd] <= t->splitEdge)
		return prRec(t->l);
	else
		return prRec(t->r);
}

double DecisionTreeClassifier::getSplitRate(vector<vector<int>>& q)
{
	double r1, r2, r3, r4;
	r1 = double(q[0][0]) / (q[0][0] + q[1][0]);
	r2 = double(q[0][1]) / (q[0][1] + q[1][1]);
	r3 = double(q[1][0]) / (q[1][0] + q[0][0]);
	r4 = double(q[1][1]) / (q[1][1] + q[0][1]);
	return max(max(max(r1, r2), r3), r4);
}

void DecisionTreeClassifier::getSplit(int &splitInd, double &splitEdge,double &maPers)
{
	for (int j = 0; j < prSize; ++j) {
		double d = (prMas[j] - prMis[j]) / (features);
		for (double k = prMis[j]; k < prMas[j]; k += d) {
			vector< vector< int > > q = vector< vector< int > >(2, vector<int>(2, 0));
			for (int i = 0; i < data.size(); ++i) {
				if (u[i]) continue;
				q[target[i]][data[i][j] > k]++;
			}
			double cr = getSplitRate(q);
			if (cr > maPers)
				maPers = cr, splitInd = j, splitEdge = k;
		}
	}
}

void DecisionTreeClassifier::processData()
{
	prSize = data[0].size();
	u.assign(data.size(), 0);
	prMis.resize(prSize);
	prMas.resize(prSize);
	for (int j = 0; j < prSize; ++j) {
		double ma = -inf, mi = inf;
		for (int i = 0; i < data.size(); ++i) {
			ma = max(ma, data[i][j]);
			mi = min(mi, data[i][j]);
		}
		prMis[j] = mi, prMas[j] = ma;
	}
	int q[2] = { 0 };
	for (int i = 0; i < target.size(); ++i) {
		q[target[i]]++;
	}
	t = new tree(q[0], q[1], 0);
}

DecisionTreeClassifier::DecisionTreeClassifier()
{
}


DecisionTreeClassifier::~DecisionTreeClassifier()
{
}



void DecisionTreeClassifier::fit(vector<vector<T>> data, vector<int> target)
{
	this->data.assign(data.begin(), data.end());
	this->target.assign(target.begin(), target.end());
	processData();
	treeRec(t);
}

int DecisionTreeClassifier::predict(vector<T> prData)
{
	this->prData.assign(prData.begin(), prData.end());
	return prRec(t);
}

void DecisionTreeClassifier::setF(int f)
{
	features = f;
}

void DecisionTreeClassifier::setMd(int md)
{
	maxDepth = md;
}

void DecisionTreeClassifier::setMn(int mn)
{
	maxNodes = mn;
}

void DecisionTreeClassifier::setLp(double lp)
{
	leafPers = lp;
}
