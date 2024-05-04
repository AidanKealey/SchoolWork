#include "client.hpp"
#include "rpc.pb.h"

#include <iostream>


void RPCClient::setServerAddress(const string& addr) {
    serverAddress = addr;
    stub.setServerAddress(addr); 
}


void RPCClient::start() {
    cout << "RPCClient starting." << endl;

    int32_t key = 1;
    string value = "Hello, RPC World!";

    try {
        bool putSuccess = stub.kvPut(key, value);
        if (putSuccess) {
            cout << "Successfully put value: " << value << " at key: " << key << endl;
        } else {
            cerr << "Failed to put value at key: " << key << endl;
        }

        auto getResult = stub.kvGet(key);
        if (getResult.first) {
            cout << "Successfully got value: " << getResult.second << " for key: " << key << endl;
        } else {
            cerr << "Failed to get value for key: " << key << endl;
        }
    } catch (const exception& e) {
        cerr << "Exception encountered during RPC operations: " << e.what() << std::endl;
    }
}

