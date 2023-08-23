const mongoose = require('mongoose');

const mongoURI = "mongodb://localhost:27017/Test"

const connectToMongo = () => {
    const con =   mongoose.connection
    con.on('open',function(){
        console.log("connection established......")
    })
}

module.exports = connectToMongo;