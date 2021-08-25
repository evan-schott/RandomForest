"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var iris_json_1 = __importDefault(require("./iris.json")); // Assume that have header with format: [param1, ..., paramN, class label]
// Associate index number with each class name
var classArr; // index => name
var classMap; // name => index
var classLabelNum; // length of the class
var dataColNum; // Number of columns in data 
var trainArr;
var validArr;
var testArr;
var NodeType;
(function (NodeType) {
    NodeType[NodeType["root"] = 0] = "root";
    NodeType[NodeType["nonTerminal"] = 1] = "nonTerminal";
    NodeType[NodeType["leaf"] = 2] = "leaf";
})(NodeType || (NodeType = {}));
var TreeNode = /** @class */ (function () {
    function TreeNode(_a) {
        var type = _a.type, dataset = _a.dataset;
        this.type = type;
        this.data = dataset;
        this.children = [];
        this.result = [];
        // Calculate gini index
        var buckets = [];
        for (var i = 0; i < classLabelNum; i++) {
            buckets.push(0);
        }
        // Set gini impurity
        this.data.forEach(function (line) {
            buckets[classMap.get(line[dataColNum - 1])]++;
        });
        this.gini = getGiniImpurity(buckets);
    }
    TreeNode.prototype.predictSample = function (sample) {
        if (this.type == NodeType.leaf) {
            return this.vote;
        }
        else {
            return this.children[this.splitter.fn(sample)].predictSample(sample);
        }
    };
    return TreeNode;
}());
var RandomForest = /** @class */ (function () {
    function RandomForest(_a) {
        var dataset = _a.dataset, featureSet = _a.featureSet, numTrees = _a.numTrees, numFeatures = _a.numFeatures;
        this.dataset = dataset;
        this.featureSet = featureSet;
        this.numTrees = numTrees;
        this.numFeatures = numFeatures;
        this.decisionTrees = [];
    }
    // Create forest of decision trees
    RandomForest.prototype.createForest = function () {
        for (var i = 0; i < this.numTrees; i++) {
            var root = new TreeNode({ type: NodeType.root, dataset: this.dataset });
            buildTree(root, randomSubSet(this.featureSet, this.numFeatures)); // Builds tree with random subset of features
            this.decisionTrees.push(root);
        }
    };
    RandomForest.prototype.ensemblePredict = function (sample) {
        var voteTallies = []; // Set up vote array
        for (var i = 0; i < classLabelNum; i++) {
            voteTallies.push(0);
        }
        this.decisionTrees.forEach(function (tree) { return voteTallies[classMap.get((tree.predictSample(sample)))]++; });
        var maxVotesIndex = 0;
        for (var i = 1; i < classLabelNum; i++) {
            if (voteTallies[i] > voteTallies[maxVotesIndex]) {
                maxVotesIndex = i;
            }
        }
        return classArr[maxVotesIndex];
    };
    RandomForest.prototype.fullTest = function (sampleSet) {
        var _this = this;
        var correct = 0;
        sampleSet.forEach(function (sample) {
            if (_this.ensemblePredict(sample) == sample[dataColNum - 1]) {
                correct++;
            }
        });
        console.log("Correct: " + String(correct) + "/" + String(sampleSet.length));
    };
    return RandomForest;
}());
function processData(trainProportion, validProportion, testProportion) {
    if ((trainProportion + validProportion + testProportion) != 1) {
        throw new RangeError();
    }
    trainArr = [];
    validArr = [];
    testArr = [];
    classMap = new Map();
    classArr = [];
    var curIndex = 0;
    dataColNum = iris_json_1.default[0].length;
    // Calculate when to switch between train, valid, test data
    var trainStop = trainProportion * iris_json_1.default.length;
    var validStop = trainStop + validProportion * iris_json_1.default.length;
    // Shuffle array before dividing it
    var shuffledData = randomSubSet(iris_json_1.default, iris_json_1.default.length);
    shuffledData.forEach(function (line, index) {
        // create copy in array format
        var curLine = [];
        line.forEach(function (elem) {
            curLine.push(elem);
        });
        if (index < trainStop) {
            trainArr.push(curLine);
        }
        else if (index < validStop) {
            validArr.push(curLine);
        }
        else {
            testArr.push(curLine);
        }
        // determine unique results
        var curClass = line[dataColNum - 1];
        if (!(classMap.has(curClass))) {
            classArr.push(curClass);
            classMap.set(curClass, curIndex++);
        }
    });
    classLabelNum = classArr.length;
}
function getRandomIntInclusive(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1) + min);
}
function randomSubSet(list, subSetLen) {
    if (list.length < 2) {
        return list;
    }
    var returnList = [];
    var listLen = list.length;
    for (var i = 0; i < subSetLen; i++) {
        var swapIndex = getRandomIntInclusive(i, listLen - 1);
        var swapValue = list[swapIndex];
        list[swapIndex] = list[i]; // Make sure each item has equal chance, and no duplicates
        returnList.push(swapValue);
    }
    return returnList;
}
function getGiniImpurity(buckets) {
    var gini = 1;
    var total = 0;
    buckets.forEach(function (bucket) { return total += bucket; });
    buckets.forEach(function (bucket) {
        gini -= Math.pow(bucket / total, 2);
    });
    return gini;
}
// Recursive function to build tree one node at a time
function buildTree(node, featureSet) {
    // See which feature best splits data
    var maxGainIndex = null;
    var maxGainValue = 0;
    featureSet.forEach(function (feature, index) {
        var buckets = []; // Use to see gini impurity of each child after hypothetical split at current feature
        for (var i = 0; i < feature.numSplit; i++) { // Initialize depending on how many categories feature divides into / how many class labels
            var cur = [];
            for (var j = 0; j < classLabelNum; j++) {
                cur.push(0);
            }
            buckets.push(cur);
        }
        node.data.forEach(function (datapoint) {
            buckets[feature.fn(datapoint)][classMap.get(datapoint[dataColNum - 1])]++;
        });
        var rawGini = 0;
        var total = 0;
        buckets.forEach(function (bucket) {
            var weight = 0;
            var gini = getGiniImpurity(bucket);
            bucket.forEach(function (labelQuant) { return weight += labelQuant; });
            rawGini += gini * weight;
            total += weight;
        });
        var totalGini = rawGini / total; // Normalize information gain
        if ((node.gini - totalGini) > maxGainValue) { // Determine if current feature results in better information gain than previous features
            maxGainValue = node.gini - totalGini;
            maxGainIndex = index;
        }
    });
    if (maxGainValue == 0) { // Case that impossible to gain more information, need to convert node to leaf
        node.type = NodeType.leaf;
        var buckets_1 = [];
        for (var i = 0; i < classLabelNum; i++) {
            buckets_1.push(0);
        }
        node.data.forEach(function (line) { return buckets_1[classMap.get(line[dataColNum - 1])]++; }); // Increment counters to see current composition of class types
        var majorityIndex = 0;
        for (var i = 1; i < classLabelNum; i++) { // Determine majority class (Who current decision tree will vote for)
            if (buckets_1[i] > buckets_1[majorityIndex]) {
                majorityIndex = i;
            }
        }
        node.vote = classArr[majorityIndex];
        buckets_1.forEach(function (bucket, index) {
            node.result.push({ name: classArr[index], percentage: bucket / (node.data.length), number: bucket });
        });
    }
    else { // Case that more information gain possible, create child node for each feature division, recursively call function on children
        var childrenData_1 = []; // Divide current dataset based on the feature splitter before give to children
        for (var i = 0; i < featureSet[maxGainIndex].numSplit; i++) {
            childrenData_1.push([]);
        }
        node.data.forEach(function (line) {
            childrenData_1[featureSet[maxGainIndex].fn(line)].push(line);
        });
        var leftoverFeatures_1 = []; // Create feature subset with remaining features for children to use
        featureSet.forEach(function (feature, index) {
            if (index != maxGainIndex) {
                leftoverFeatures_1.push(feature);
            }
            else {
                node.splitter = feature;
            }
        });
        childrenData_1.forEach(function (newChildData, index) {
            var child = new TreeNode({ type: NodeType.nonTerminal, dataset: newChildData });
            node.children.push(child); // link child to current node
            buildTree(child, leftoverFeatures_1);
        });
    }
}
var featureSetTest = [
    {
        name: "sepal length",
        fn: function (params) {
            if (params[0] < 5) {
                return 0;
            }
            else if (params[0] < 6) {
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
        fn: function (params) {
            if (params[1] < 3) {
                return 0;
            }
            else {
                return 1;
            }
        },
        numSplit: 2
    },
    {
        name: "petal length",
        fn: function (params) {
            if (params[2] < 2) {
                return 0;
            }
            else if (params[2] < 6) {
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
        fn: function (params) {
            if (params[3] < 1) {
                return 0;
            }
            else if (params[3] < 2) {
                return 1;
            }
            else {
                return 2;
            }
        },
        numSplit: 3
    }
];
processData(.7, 0, .3); // Read data to get basic info
// Train the random forest
var rf = new RandomForest({ dataset: trainArr, featureSet: featureSetTest, numTrees: 100, numFeatures: Math.sqrt(featureSetTest.length) });
rf.createForest();
// Test the random forest
rf.fullTest(testArr);
//console.log(rf.ensemblePredict([5.1,3.5,1.4,0.2,"setosa"]));
//console.log(rf.decisionTrees[0].predictSample([5.1,3.5,1.4,0.2,"setosa"]));
console.log("done");
//# sourceMappingURL=index.js.map