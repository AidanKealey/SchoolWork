#include "kvserver.hpp"


KVServer::KVServer(string nodeName): Node(nodeName) {
    kvService = make_shared<KVServiceServer>(nodeName, weak_from_this());
    addService(kvService);
}

void KVServer::setDBMFileName(string name) {
    kvService->setDBMFileName(name);
}

void KVServer::setNickname(string nickname) {
    kvService->setNickname(nickname);
}

void KVServer::setPort(in_port_t PORT) {
    kvService->setPort(PORT);
}

void KVServer::setIsPrimary(bool isPrimary) {
    kvService->setIsPrimary(isPrimary);
}

void KVServer::addBackupName(string address) {
    // These methods only make sense for the primary server
    if (kvService->isPrimary) {
        kvService->addBackupName(address);
    }
}

void KVServer::addBackupPort(in_port_t port) {
    // These methods only make sense for the primary server
    if (kvService->isPrimary) {
        kvService->addBackupPort(port);
    }
}


void KVServer::setPrimaryServer(string name) {
    // This method is for backup servers to set their primary server's name
    if (!kvService->isPrimary) {
        kvService->setPrimaryServer(name);
    }
}
