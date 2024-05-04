#include <iostream>

#include "sdservice.hpp"

#ifdef __APPLE__
#define MSG_CONFIRM 0
#endif

using namespace std;
using namespace string_literals;

#define close mclose
void mclose(int fd);


void SDService::stop() {
    alive = false;
}


void SDService::start() {
    //cerr << "in SDService::start" << endl;
    struct sockaddr_in servaddr, cliaddr;

    // get a socket to recieve messges
    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) {
        perror("socket creation failed");
        return; // this will exit the service thread and stop the server
    }

    // clear variables before initializing
    memset(&servaddr, 0, sizeof(servaddr));
    memset(&cliaddr, 0, sizeof(cliaddr));

    // Port and Interfact information for binding
    // the socket
    servaddr.sin_family = AF_INET;        // IPv4
    servaddr.sin_addr.s_addr = INADDR_ANY;   // whatever interface is available
    servaddr.sin_port = htons(PORT);

    // cout << "DEBUG: sdservice::start() port num: " << PORT << " htons(PORT): " << htons(PORT) << endl;

    // Bind the socket with the server address
    if (::bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0 )
    {
        perror("bind failed");
        return; // this will exit the service thread and stop the server
    }

    socklen_t len;
    int n;

    while(alive) {
	    len = sizeof(cliaddr);  //len is value/result
        n = recvfrom(sockfd, (uint8_t *)udpMessage, MAXMSG, 
		    MSG_WAITALL, (struct sockaddr *) &cliaddr,
		    &len);

        if(n > 0) {
            cout << "SDService::start() successful recieved in while" << endl;
        } else {
            cout << "SDService::start() bad recieved in while" << endl;
        }

        E477DirectoryService::DSRequest receivedMsg;
        E477DirectoryService::DSResponse replyMsg;

        if (!receivedMsg.ParseFromArray(udpMessage, n)){
            cerr << "Could not parse message" << endl;
            // ignore
	    }

        if((receivedMsg.magic()) != magic) {
            cerr << "service unrecognized message" << endl;
        } else {
            // start by copying version and serial to reply
            replyMsg.set_magic(magic);
            replyMsg.set_version(receivedMsg.version());
            replyMsg.set_serial(receivedMsg.serial());

            if(receivedMsg.version() == version1x) {
                queryDirectory(receivedMsg, replyMsg);
            } else {
                cerr << "unrecognized version" << endl;
                // For now ignore, message doesn't have a wrong version reply
            }

            // at this point in time the reply is complete
            // send response back
            uint32_t msglen = replyMsg.ByteSizeLong();
            // double check size
            replyMsg.SerializeToArray(udpMessage, msglen);
            //cerr << "reply message" << HexDump{udpMessage,msglen} ;

            int servern = sendto(sockfd, udpMessage, msglen,
                MSG_CONFIRM, (const struct sockaddr *) &cliaddr, len);
        }
    }
    
    close(sockfd);
}


void SDService::queryDirectory(E477DirectoryService::DSRequest &receivedMsg, E477DirectoryService::DSResponse &replyMsg) {
    // registering new server
    if(receivedMsg.has_registerserviceargs()) {
        cerr << "SDService: registering new server in directory" << endl;

        const E477DirectoryService::registerServiceRequest &registerReq = receivedMsg.registerserviceargs();

        string nickname = registerReq.servicename();

        struct directoryRecord aRecord;
        aRecord.server = registerReq.servername();
        aRecord.port = registerReq.port();

        // cerr << nickname << endl;
        // cerr << aRecord.server << endl;
        // cerr << aRecord.port << endl;
        
        bool registerStatus = registerServer(nickname, aRecord);

        if(registerStatus) {
            cerr << "registration -- COMPLETE" << endl;
        } else {
            cerr << "registration -- FAILED" << endl;
        }

        E477DirectoryService::registerServiceResponse *registerResponse = replyMsg.mutable_registerserviceres();
        registerResponse->set_success(registerStatus);
    }
    
    // searching for server
    if(receivedMsg.has_searchserviceargs()) {
        cerr << "SDService: searching for server in directory" << endl;

        const E477DirectoryService::searchServiceRequest &searchReq = receivedMsg.searchserviceargs();

        string name = searchReq.servicename();

        struct directoryRecord searchedRecord = searchForService(name);

        bool inDirectory = false;

        if(searchedRecord.server != "" && searchedRecord.port != 0) {
            cerr << "search results good, found server - " << searchedRecord.server << " and port - " << searchedRecord.port << endl;
            inDirectory = true;
        } else {
            cerr << "couldn't find server" << endl;
            inDirectory = false;
        }

        E477DirectoryService::searchServiceResponse *searchResponse = replyMsg.mutable_searchserviceres();
        searchResponse->set_found(inDirectory);
        searchResponse->set_servername(searchedRecord.server);
        searchResponse->set_port(searchedRecord.port);

        // cerr << "successful searching" << endl;
    }

    // deleting server
    if(receivedMsg.has_deleteserviceargs()) {
        cerr << "SDService: deleting server in directory" << endl;

        const E477DirectoryService::deleteServiceRequest &deleteReq = receivedMsg.deleteserviceargs();

        string name = deleteReq.servicename();

        bool deleteStatus = deleteServer(name);

        if(deleteStatus) {
            cerr << "deleted server" << endl;
        } else {
            cerr << "failed to server" << endl;
        }

        E477DirectoryService::deleteServiceResponse *deleteResponse = replyMsg.mutable_deleteserviceres();
        deleteResponse->set_success(deleteStatus);
    }

    return;
}


bool SDService::registerServer(const string &name, struct directoryRecord aRecord) {
    //cerr << "in SDService::registerServer" << endl;
    bool insertRecord = false;
    int sizebBefore = this->directory.size();
    this->directory.insert({name, aRecord});
    int sizeAfter = this->directory.size();

    if(sizeAfter > sizebBefore) {
        insertRecord = true;
    } else {
        insertRecord = false;
    }

    return insertRecord;
}


struct directoryRecord SDService::searchForService(string nickname) {
    cerr << "in SDService::searchForService looking for: " << nickname << endl;

    struct directoryRecord searchRecord;
    bool searchStatus = false;
    auto searchResults = this->directory.find(nickname);

    if(searchResults != this->directory.end()) {
        searchStatus = true;
        searchRecord = searchResults->second;
        cerr << "service for server (" << nickname << ") found in directory" << endl;
    } else {
        searchStatus = false;
        searchRecord.server = "";
        searchRecord.port = 0;
        cerr << "serivce for server (" << nickname << ") not found in directory" << endl;
    }

    return searchRecord;
}


bool SDService::deleteServer(string nickname) {
    bool successfulDelete = this->directory.erase(nickname);
    
    return successfulDelete;
}
