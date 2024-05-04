#ifndef __RPC_CLIENT_HPP__
#define __RPC_CLIENT_HPP__

#include "network.hpp"
#include "clientStub.hpp"
#include <string>

using namespace std;

class RPCClient : public Node {
    private:
        ClientStub stub; 
        string serverAddress; 

    public:
        RPCClient(string nodeName) : Node(nodeName) {}
        ~RPCClient() {}

        void setServerAddress(const string& addr);
        void start();
};

#endif
