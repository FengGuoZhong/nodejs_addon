var addon = require('./build/Release/neurons.node');

var data = "1,7,2,44,45,55,56,57,58,59";
console.log( 'Learn:', addon.Learn(data) );


data = "0,3,44,45,55";
console.log( 'Classify:', addon.Classify(data) );