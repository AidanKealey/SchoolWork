#ifndef __SERVER_HPP__
#define __SERVER_HPP__

#include "network.hpp"
#include "service.hpp"
#include <string>
#include <gdbm.h>

using namespace std;

class RPCServer : public Node {
    private:
        shared_ptr<RPCService> theService;
        
    public:
        RPCServer(string nodeName);
        ~RPCServer() {};

        void setDatabaseName(const string& name); 

};

#endif