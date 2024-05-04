
#include "kvserver.hpp"


KVServer::KVServer(string nodeName): Node(nodeName){
            // note this is the same as installng the service software on the server
            // it doesn't run yet until the node is started.

            //cout << "Main: Server " << nodeName << " adding key value service" << endl;

	    kvService = make_shared<KVServiceServer>(nodeName,weak_from_this());

            addService(kvService);
        }

// void KVServer::start(){
//    //cerr << "in KVSERVER start" << endl;
// }


void KVServer::setDBMFileName(string name){
    kvService -> setDBMFileName(name);
}


void KVServer::setNickname(string nickname){
    kvService -> setNickname(nickname);
}
        

void KVServer::setPort(in_port_t PORT) {
    kvService ->setPort(PORT);
}

