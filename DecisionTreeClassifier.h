#pragma once
#include <vector>
#include <Windows.h>
#define T double
using namespace std;
struct tree {
	int claSs, depth, fq, mq, splitInd;
	double splitEdge;
	tree *l, *r;
	tree(int fq, int mq, int depth) :l(NULL), r(NULL), claSs(-1),
		fq(fq), mq(mq), depth(depth) {}
};
class DecisionTreeClassifier
{
private:
	vector< vector<double> > data;
	vector<double> prMas, prMis,  prData;
	vector<int> target, u;
	int prSize;
	int features = 30;
	int maxDepth = 18;
	int maxNodes = (1 << 13), nodes=0;
	double leafPers = 0.95;
	tree *t;
	void treeRec(tree*);
	int prRec(tree*);
	double getSplitRate(vector< vector< int > >&);
	void getSplit(int&, double&, double&);
	void processData();
public:
	DecisionTreeClassifier();
	~DecisionTreeClassifier();
	void fit(vector< vector<T> >,vector<int>);
	int predict(vector<T>);
	void setF(int);
	void setMd(int);
	void setMn(int);
	void setLp(double);
};

