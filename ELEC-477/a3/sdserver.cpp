#include "sdserver.hpp"


SDServer::SDServer(string nodeName): Node(nodeName){
    sdService = make_shared<SDService>(nodeName,weak_from_this());
    addService(sdService);
}

void SDServer::start() {
    cout << "Service Directory Server starting" << endl;
}