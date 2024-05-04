#include "server.hpp"

RPCServer::RPCServer(string nodeName) : Node(nodeName) {
    cout << "Main: Server " << nodeName << " adding rpc service" << endl;
    theService = make_shared<RPCService>(nodeName, weak_from_this());
    
    setDatabaseName(nodeName+".gdbm");
    addService(theService);
}

void RPCServer::setDatabaseName(const string& name) {
    theService->setDatabaseName(name);
}
