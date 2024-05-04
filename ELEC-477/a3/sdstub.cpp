#include "sdstub.hpp"

#define close mclose
void mclose(int fd);

using namespace std;

bool SDStub::init() {
    //cerr << "in SDStub init" << endl;

    memset(&serveraddr, 0, sizeof(serveraddr));
    serveraddr.sin_family = AF_INET;
    serveraddr.sin_port = htons(PORT);

    struct addrinfo *res;
    int numAddr = getaddrinfo(serverName.c_str(), nullptr, nullptr, &res);
    serveraddr.sin_addr = ((struct sockaddr_in*)res -> ai_addr) -> sin_addr;
    freeaddrinfo(res);

    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) {
	perror("socket creation failed");
        return false;
    }

    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv,sizeof(tv)) < 0) {
        perror("Error");
        return false;
    }

    ready = true;
    return true;

}


bool SDStub::registerNewServer(const string &nickname, struct directoryRecord aRecord) {
    cerr << "SDStub: attempting to register new service" << endl;

    if (!ready){
        if (!init()) {
            cerr << "!init" << endl;
            // init returned false, so problem setting up the
            // socket onnection
            return false;
        }
    }

    struct sockaddr_in serveraddrreply;

    int n;
    socklen_t len;
    uint32_t blen = MAXMSG;
    uint8_t buffer[MAXMSG];
    uint32_t serial = this->serial++;

    E477DirectoryService::DSRequest requestMsg;
    requestMsg.set_magic(magic);
    requestMsg.set_version(version1x);
    requestMsg.set_serial(serial++);
    
    E477DirectoryService::registerServiceRequest *registerService = requestMsg.mutable_registerserviceargs();
    registerService->set_servicename(nickname);
    registerService->set_servername(aRecord.server);
    registerService->set_port(aRecord.port);

    // cerr << nickname << endl;
    // cerr << aRecord.server << endl;
    // cerr << aRecord.port << endl;

    blen = requestMsg.ByteSizeLong();
    if(blen > MAXMSG) {
        cerr << "message is too long" << endl;
        return false;
    }

    if(!requestMsg.SerializeToArray(buffer, blen)) {
        cerr << "failed to serialize request" << endl;
        return false;
    }

    n = sendto(sockfd, (const char *)buffer, blen, 0, 
        (const struct sockaddr *) &serveraddr, sizeof(serveraddr));

    // cout << "DEBUG: sending to serveraddr port num: " << PORT << " htons(PORT) is: " << htons(PORT) <<endl;

    // cerr << "Sending " << blen << "bytes to address " << ntoa(serveraddr.sin_addr.s_addr) << endl;
    // cerr << "Sending " << blen << "bytes to address " << inet_ntoa(serveraddr.sin_addr) << endl;
    if(n < 0) {
        cerr << "SDStub: failed to send to socket: " << sockfd << endl;
        return false;
    } else {
        cerr << "SDStub: successful send to socket: " << sockfd << endl;
    }

    E477DirectoryService::DSResponse responseMsg;
    bool recievedResponse = true;
    do {
        len = sizeof(struct sockaddr_in);
        // cerr << "recv " << len << "bytes from address " << inet_ntoa(serveraddrreply.sin_addr) << endl;
        n = recvfrom(sockfd, (char *) buffer, MAXMSG,
                    MSG_WAITALL, (struct sockaddr *) &serveraddrreply, &len);
        // cerr << "recv " << len << "bytes from address " << inet_ntoa(serveraddrreply.sin_addr) << endl;
        
        if(n == -1) {return false;}

        if(!responseMsg.ParseFromArray(buffer, n)) {
            cerr << "could not parse mnessage" << endl;
        } else {
            if(requestMsg.version() != responseMsg.version()) {
                cerr << "version mismatch" << endl;
                recievedResponse = false;
            } else {
                if(requestMsg.serial() != responseMsg.serial()) {
                    cerr << "serial numbers mismatch" << endl;
                    recievedResponse = false;
                } else {
                    if(responseMsg.has_registerserviceres()) {
                        E477DirectoryService::registerServiceResponse registerResponse = responseMsg.registerserviceres();
                        // cerr << "registerResponse.success();" << registerResponse.success() << endl;
                        return registerResponse.success();
                    }
                }
            }
        }
    } while(!recievedResponse);

    cerr << "should not make it here" << endl;

    return 0;
}


directoryRecord SDStub::searchForService(string serviceName) {
    cerr << "SDStub::searchForService" << endl;
    cerr << "attempting to search for service" << endl;
    cerr << "service nickname: " << serviceName << endl;
    directoryRecord aRecord;
    aRecord.server = "";
    aRecord.port = 0;
    if (!ready){
        if (!init()) {
            cerr << "!init" << endl;
            // init returned false, so problem setting up the
            // socket onnection
            
            return aRecord;
        }
    }

    struct sockaddr_in serveraddrreply;

    int n;
    socklen_t len;
    uint32_t blen = MAXMSG;
    uint8_t buffer[MAXMSG];
    bool searchMsgStatus = true; 

    E477DirectoryService::DSRequest requestMsg;
    requestMsg.set_magic(magic);
    requestMsg.set_version(version1x);
    requestMsg.set_serial(serial++);

    E477DirectoryService::searchServiceRequest *searchService = requestMsg.mutable_searchserviceargs();
    searchService->set_servicename(serviceName);

    blen = requestMsg.ByteSizeLong();
    if(blen > MAXMSG) {
        cerr << "message is too long" << endl;
        searchMsgStatus = false;
    }

    if(!requestMsg.SerializeToArray(buffer, blen)) {
        cerr << "failed to serialize request" << endl;
        searchMsgStatus = false;
    }

    n = sendto(sockfd, (const char *)buffer, blen, 0, 
        (const struct sockaddr *) &serveraddr, sizeof(serveraddr));

    // cout << "DEBUG: sending to serveraddr port num: " << PORT << " htons(PORT) is: " << htons(PORT) <<endl;

    // cerr << "Sending " << blen << "bytes to address " << ntoa(serveraddr.sin_addr.s_addr) << endl;
    // cerr << "Sending " << blen << "bytes to address " << inet_ntoa(serveraddr.sin_addr) << endl;
    if(n < 0) {
        cerr << "failed to send to socket: " << sockfd << endl;
        return aRecord;
    } else {
        cerr << "successful send to socket: " << sockfd << endl;
    }

    E477DirectoryService::DSResponse responseMsg;
    bool recievedResponse = true;
    bool recievedResponseStatus = false;
    if(searchMsgStatus) {
        do {
            len = sizeof(struct sockaddr_in);
            n = recvfrom(sockfd, (char *) buffer, MAXMSG,
                        MSG_WAITALL, (struct sockaddr *) &serveraddrreply, &len);
            
            // cerr << "recv " << blen << "bytes from address " << inet_ntoa(serveraddrreply.sin_addr) << endl;

            if(n == -1) {
                recievedResponseStatus = false;
                break;
            }

            if(!responseMsg.ParseFromArray(buffer, n)) {
                cerr << "could not parse mnessage" << endl;
            } else {
                if(requestMsg.version() != responseMsg.version()) {
                    cerr << "version mismatch" << endl;
                    recievedResponseStatus = false;
                    recievedResponse = false;
                } else {
                    if(requestMsg.serial() != responseMsg.serial()) {
                        cerr << "serial numbers mismatch" << endl;
                        recievedResponseStatus = false;
                        recievedResponse = false;
                    } else {
                        if(responseMsg.has_searchserviceres()) {
                            recievedResponseStatus = true;
                            recievedResponse = true;
                        }
                    }
                }
            }
        } while(!recievedResponse);
    } else {
        cerr << "search request issue" << endl;
    }

    E477DirectoryService::searchServiceResponse searchServiceRes = responseMsg.searchserviceres();

    directoryRecord response;

    if(recievedResponseStatus) {
        cerr << "search message received: good" << endl;
        response.server = searchServiceRes.servername();
        response.port = searchServiceRes.port();
        
        return response;
    } else {
        cerr << "search response issue" << endl;
        response.server = "";
        response.port = 0;
        
        return response;
    }
}

bool SDStub::deleteServer(string serviceName) {
    cerr << "atempting to delete a service" << endl;
    cerr << "is ready? " << ready << endl;
    if (!ready){
        if (!init()) {
            cerr << "!init" << endl;
            // init returned false, so problem setting up the
            // socket onnection
            return 0;
        }
    }

    struct sockaddr_in serveraddrreply;

    int n;
    socklen_t len;
    uint32_t blen = MAXMSG;
    uint8_t buffer[MAXMSG]; 

    E477DirectoryService::DSRequest requestMsg;
    requestMsg.set_magic(magic);
    requestMsg.set_version(version1x);
    requestMsg.set_serial(serial++);

    E477DirectoryService::deleteServiceRequest *deleteService = requestMsg.mutable_deleteserviceargs();
    deleteService->set_servicename(serviceName);

    blen = requestMsg.ByteSizeLong();
    if(blen > MAXMSG) {
        cerr << "message is too long" << endl;
        return false;
    }

    if(!requestMsg.SerializeToArray(buffer, blen)) {
        cerr << "failed to serialize request" << endl;
        return false;
    }

    if(sendto(sockfd, buffer, blen, 0, 
        (const struct sockaddr *) &serveraddr, sizeof(serveraddr))) {
        cerr << "failed to send to socket: " << sockfd << endl;
        return false;
    } else {
        cerr << "successful send to socket: " << sockfd << endl;
    }

    E477DirectoryService::DSResponse responseMsg;
    bool recievedResponse = true;
    do {
        len = sizeof(struct sockaddr_in);
        n = recvfrom(sockfd, (char *) buffer, MAXMSG,
                    MSG_WAITALL, (struct sockaddr *) &serveraddrreply, &len);\
        
        if(n == -1) {return false;}

        if(!responseMsg.ParseFromArray(buffer, n)) {
            cerr << "could not parse mnessage" << endl;
        } else {
            if(requestMsg.version() != responseMsg.version()) {
                cerr << "version mismatch" << endl;
                recievedResponse = false;
            } else {
                if(requestMsg.serial() != responseMsg.serial()) {
                    cerr << "serial numbers mismatch" << endl;
                    recievedResponse = false;
                } else {
                    if(responseMsg.has_deleteserviceres()) {
                        E477DirectoryService::registerServiceResponse registerResponse;
                        return registerResponse.success();
                    }
                }
            }
        }
    } while(!recievedResponse);

    return 0;
}