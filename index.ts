import data from "./iris.json"; // Assume that have header with format: [param1, ..., paramN, class label]

// Associate index number with each class name
let classArr; // map from [index] => [name]
let classMap; // map from [name] => [index]

let classLabelNum; // length of the class
let dataColNum; // Number of columns in data 

let trainArr;
let validArr;
let testArr;

enum NodeType {
    root,
    nonTerminal,
    leaf
}

type feature = {
    name: String,
    fn: (params: (number | string)[]) => number,
    numSplit: number
};

type leafResult = {
    name: String,
    percentage: number,
    number: number
};

class TreeNode {
    type: NodeType;
    children: TreeNode[];
    data: (number | string)[][];
    vote: string;
    result: leafResult[];
    gini: number;
    splitter: feature;

    predictSample(sample:(number | string)[]):string {
        if (this.type == NodeType.leaf) {
            return this.vote;
        }
        else {
            return this.children[this.splitter.fn(sample)].predictSample(sample);
        }
    }

    constructor({type, dataset}) {
        this.type = type;
        this.data = dataset;
        this.children = [];
        this.result = [];

        // Calculate gini index
        let buckets: number[] = [];
        for (let i = 0; i < classLabelNum; i++) {
            buckets.push(0);
        }

        // Set gini impurity
        this.data.forEach((line) => {
            buckets[classMap.get(line[dataColNum - 1])]++;
        });
        this.gini = getGiniImpurity(buckets);
    }
}

class RandomForest {
    dataset: ((number | string)[])[];
    featureSet: feature[];
    decisionTrees: TreeNode[];

    // Parameters to be set according to other inputs
    numTrees: number;
    numFeatures: number;

    constructor({dataset, featureSet, numTrees, numFeatures}) {
        this.dataset = dataset;
        this.featureSet = featureSet;
        this.numTrees = numTrees;
        this.numFeatures = numFeatures;
        this.decisionTrees = [];
    }

    // Create forest of decision trees
    createForest() {
        for (let i = 0; i < this.numTrees; i++) { 
            let root: TreeNode = new TreeNode({type: NodeType.root, dataset: this.dataset});
            buildTree(root, randomSubSet(this.featureSet, this.numFeatures)); // Builds tree with random subset of features
            this.decisionTrees.push(root);
        }
    }

    ensemblePredict(sample: (number | string)[]):string {
        let voteTallies:number[] = []; // Set up vote array
        for (let i = 0; i < classLabelNum; i++) {
            voteTallies.push(0);
        }

        this.decisionTrees.forEach(tree => voteTallies[classMap.get((tree.predictSample(sample)))]++);
        let maxVotesIndex = 0;
        for (let i = 1; i < classLabelNum; i++){
            if (voteTallies[i] > voteTallies[maxVotesIndex]) {
                maxVotesIndex = i;
            }
        }

        return classArr[maxVotesIndex];
    }

    fullTest(sampleSet: (number | string)[][]) {
        let correct = 0;
        sampleSet.forEach((sample) => {
            if (this.ensemblePredict(sample) == sample[dataColNum - 1]){
                correct++;
            }
        })
        console.log("Correct: " + String(correct) + "/" + String(sampleSet.length));
    }
}


function processData(trainProportion: number, validProportion: number, testProportion: number) {

    if ((trainProportion + validProportion + testProportion) != 1) {
        throw new RangeError();
    }

    trainArr = [];
    validArr = [];
    testArr = [];
    classMap = new Map();
    classArr = [];

    let curIndex = 0;
    dataColNum = data[0].length;
    
    // Calculate when to switch between train, valid, test data
    let trainStop = trainProportion*data.length;
    let validStop = trainStop + validProportion*data.length;

    // Shuffle array before dividing it
    let shuffledData = randomSubSet(data,data.length);

    shuffledData.forEach((line, index) => {

        // create copy in array format
        let curLine: (number | string)[] = [];
        line.forEach((elem) => {
            curLine.push(elem);
        });

        if (index < trainStop){
            trainArr.push(curLine);
        }
        else if (index < validStop) {
            validArr.push(curLine);
        }
        else {
            testArr.push(curLine);
        }

        // determine unique results
        let curClass = line[dataColNum - 1];
        if (!(classMap.has(curClass))) {
            classArr.push(curClass);
            classMap.set(curClass, curIndex++);
        }
    })

    classLabelNum = classArr.length;
}

function getRandomIntInclusive(min:number, max:number):number {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1) + min);
  }

function randomSubSet(list: any[], subSetLen: number):any[] {
    if (list.length < 2) {
        return list;
    }

    let returnList = [];
    let listLen = list.length;
    for (let i = 0; i < subSetLen; i++) {
        let swapIndex = getRandomIntInclusive(i, listLen - 1);
        let swapValue = list[swapIndex];
        list[swapIndex] = list[i]; // Make sure each item has equal chance, and no duplicates
        returnList.push(swapValue);
    }

    return returnList;
}

function getGiniImpurity(buckets: number[]):number {
    let gini = 1;
    let total = 0;
    buckets.forEach(bucket => total += bucket);
    buckets.forEach(bucket => {
        gini -= Math.pow(bucket/total, 2);
    });
    
    return gini;
}

// Recursive function to build tree one node at a time
function buildTree(node: TreeNode, featureSet: feature[]){

    // See which feature best splits data
    let maxGainIndex = null;
    let maxGainValue = 0;
    
    featureSet.forEach((feature, index) => {    // See which feature results in best information gain
    
        let buckets: number[][] = []; // Use to see gini impurity of each child after hypothetical split at current feature
        
        for (let i = 0; i < feature.numSplit; i++) { // Initialize depending on how many categories feature divides into / how many class labels
            let cur: number[] = [];
            for (let j = 0; j < classLabelNum; j++){
                cur.push(0);
            }
            buckets.push(cur);
        }
        
        node.data.forEach((datapoint) => { // Loop through data, increment bucket counters 
            buckets[feature.fn(datapoint)][classMap.get(datapoint[dataColNum - 1])]++;
        });
        
        let rawGini = 0;
        let total = 0;
        buckets.forEach(bucket => { // calculate raw (not normalized yet) gini impurity for choosing this feature
            let weight = 0;
            let gini = getGiniImpurity(bucket);


            bucket.forEach(labelQuant => weight += labelQuant);
            rawGini += gini*weight;
            total += weight;
        })

        let totalGini = rawGini/total; // Normalize information gain

        if ((node.gini - totalGini) > maxGainValue) { // Determine if current feature results in better information gain than previous features
            maxGainValue = node.gini - totalGini;
            maxGainIndex = index;
        }
    });

    if (maxGainValue == 0) {    // Case that impossible to gain more information, need to convert node to leaf
        
        node.type = NodeType.leaf;

        let buckets:number[] = [];
        for (let i = 0; i < classLabelNum; i++) {
            buckets.push(0);
        }

        node.data.forEach(line => buckets[classMap.get(line[dataColNum - 1])]++); // Increment counters to see current composition of class types

        let majorityIndex = 0; 
        for (let i = 1; i < classLabelNum; i++) {    // Determine majority class (Who current decision tree will vote for)
            if (buckets[i] > buckets[majorityIndex]) {
                majorityIndex = i;
            }
        }
        node.vote = classArr[majorityIndex];

        
        buckets.forEach((bucket, index) => { // Determine result (percentage, majority, etc) for debugging purposes
            node.result.push({name: classArr[index], percentage: bucket/(node.data.length), number: bucket});
        })

       
    }
    else {  // Case that more information gain possible, create child node for each feature division, recursively call function on children


        let childrenData: (number | string)[][][] = []; // Divide current dataset based on the feature splitter before give to children
        for (let i = 0; i < featureSet[maxGainIndex].numSplit; i++) { 
            childrenData.push([]);
        }
        node.data.forEach(line => {
            childrenData[featureSet[maxGainIndex].fn(line)].push(line);
        });

        let leftoverFeatures:feature[] = []; // Create feature subset with remaining features for children to use
        featureSet.forEach((feature, index) => {
            if (index != maxGainIndex) {
                leftoverFeatures.push(feature);
            }
            else {
                node.splitter = feature;
            }
        })

        childrenData.forEach((newChildData, index) => { // Recursively call function on one child at a time
        
            let child = new TreeNode({type: NodeType.nonTerminal, dataset: newChildData});
            node.children.push(child); // link child to current node

            buildTree(child, leftoverFeatures);
        })
    }


    
}

// This is just an example that is hardcoded in. In future will have function to automatically generate best split for features, not manual split. 
let featureSetTest: feature[] = [
    {
        name: "sepal length",
        fn: (params) => {
            if (params[0] < 5)
            {
                return 0;
            }
            else if (params[0] < 6)
            {
                return 1;
            }
            else {
                return 2;
            }
        },
        numSplit: 3
    },
    {
        name: "sepal width",
        fn: (params) => {
            if (params[1] < 3)
            {
                return 0;
            }
            else 
            {
                return 1;
            }
        },
        numSplit: 2
    },
    {
        name: "petal length",
        fn: (params) => {
            if (params[2] < 2)
            {
                return 0;
            }
            else if (params[2] < 6)
            {
                return 1;
            }
            else {
                return 2;
            }
        },
        numSplit: 3
    },
    {
        name: "petal width",
        fn: (params) => {
            if (params[3] < 1)
            {
                return 0;
            }
            else if (params[3] < 2)
            {
                return 1;
            }
            else {
                return 2;
            }
        },
        numSplit: 3
    }
]


processData(.7,0,.3); // Read data to get basic info

// Train the random forest
let rf = new RandomForest({dataset: trainArr, featureSet: featureSetTest, numTrees: 100, numFeatures: Math.sqrt(featureSetTest.length)});
rf.createForest();

// Test the random forest
rf.fullTest(testArr);  