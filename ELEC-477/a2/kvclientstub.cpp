#include "kvclientstub.hpp"

using namespace std;
using namespace std::string_literals;

#define close mclose
void mclose(int fd);

bool KVServiceStub::kvPut(int32_t key,const uint8_t* value, uint16_t vlen){
    // init if needed
    if (!ready){
        if (!init()) {
            // init returned false, so problem setting up the
            // socket onnection
            return false;
        }
    }

    int n;
    socklen_t len;
    uint32_t blen = MAXMSG;
    uint8_t buffer[MAXMSG];	 // to serialize into
    struct sockaddr_in	 servaddrreply;

    // get the current value of serial for this request.
    uint32_t serial = this->serial++;

    // marshal parameters to send.
    E477KV::kvRequest msg;
    msg.set_magic(magic);
    msg.set_version(version1x);
    msg.set_serial(serial++);
    E477KV::putRequest * pr = msg.mutable_putargs();
    pr -> set_key(key);
    pr -> set_value(std::string((const char*)value, vlen));
    blen = msg.ByteSizeLong();
    if (blen > MAXMSG){
        // too long??
	std::cerr << " *** msg too long" << std::endl;
	// errno = ???
	return false;
    }
    msg.SerializeToArray(buffer, blen);
    //std::cout << HexDump{buffer,blen} << endl;

    // send the message to the server.
    //std::cout << "blen = " << dec << blen << endl;

    // search here 

    n = sendto(sockfd, (const char *)buffer, blen,
		MSG_CONFIRM, (const struct sockaddr *) &servaddr, sizeof(servaddr));
    //std::cout << "client sendto n = " << n << endl;

    // wait for the reply

    E477KV::kvResponse putRespMsg;
    bool gotMessage = true;
    do {
        len = sizeof(struct sockaddr_in);
        n = recvfrom(sockfd, (char *)buffer, MAXMSG,
                     MSG_WAITALL, (struct sockaddr *) &servaddrreply, &len);
        
        //std::stringstream ss;
        //ss << "client recieved = " << n << std::endl;
        //std::cout << ss.str();
        
        // check for timeout here..
        if (n==-1) { return false; };
        
        if (!putRespMsg.ParseFromArray(buffer,n)){
            cerr << "Couild not parse message" << endl;
            // wait for another mesage
        } else {
            if (putRespMsg.magic() != 'E477'){
                cerr << "Magic Mismatch" << endl;
                gotMessage = false;
            } else {
                if (msg.version() != putRespMsg.version()){
                    cerr << "Version Mismatch" << endl;
                    gotMessage = false;
                } else {
                    // wait for another message is the serial number is wrong.
                    if (msg.serial() != putRespMsg.serial()){
                        cerr << "Serial Numnbers Mismatch" << endl;
                        gotMessage = false;
                    }
                }
            }
        }
    } while (!gotMessage);
    
    bool returnRes = false;
    if (putRespMsg.has_putres()){
      returnRes = putRespMsg.putres().status();
    } else {
      cerr << "wrong message type" << endl;
	returnRes  = false;
    }

    return returnRes;
}

kvGetResult KVServiceStub::kvGet(int32_t key){
    // init if needed
    kvGetResult retVal;
    retVal.status=false;
    retVal.value = nullptr;
    retVal.vlen = 0;
    if (!ready){
        if (!init()) {
            // init returned false, so problem setting up the
            // socket onnection
            return retVal;
        }
    }

    struct sockaddr_in servaddrreply;

    int n;
    socklen_t len;
    uint32_t blen = MAXMSG;
    uint8_t buffer[MAXMSG];
    E477KV::kvRequest msg;
    msg.set_magic(magic);
    msg.set_version(version1x);
    msg.set_serial(serial++);
    E477KV::getRequest * gr = msg.mutable_getargs();

    gr -> set_key(key);
    blen = msg.ByteSizeLong();
    if (blen > MAXMSG){
        // too long??
        std::cerr << " *** msg too long" << std::endl;
        // errno = ???
        retVal.status = false;
        return retVal;
    }
    msg.SerializeToArray(buffer, blen);
    //std::cout << HexDump{buffer,blen} << endl;

    // send the message3
    //std::cout << "blen = " << dec << blen << endl;

    // cout << "DEBUG: sending to serveraddr port num: " << PORT << " htons(PORT) is: " << htons(PORT) <<endl;

    n = sendto(sockfd, (const char *)buffer, blen,
		MSG_CONFIRM, (const struct sockaddr *) &servaddr, sizeof(servaddr));
    //std::cout << "client sendto n = " << n << endl;

    E477KV::kvResponse getRespMsg;
    bool gotMessage = true;
    do {
        len = sizeof(struct sockaddr_in);
        n = recvfrom(sockfd, (char *)buffer, MAXMSG,
                     MSG_WAITALL, (struct sockaddr *) &servaddrreply, &len);
        
        //std::stringstream ss;
        //ss << "client recieved = " << n << std::endl;
        //std::cout << ss.str();
        //std::cout << HexDump{buffer,(uint32_t)n} << endl;
        // check for timeout here..
        if (n==-1) { return retVal; }// null ptrs and falls status };
        
        if (!getRespMsg.ParseFromArray(buffer,n)){
            cerr << "Couild not parse message" << endl;
            // wait for another mesage
        } else {
            if (getRespMsg.magic() != magic){
                gotMessage = false;
            } else {
                if (msg.version() != getRespMsg.version()){
                    cerr << "Version Mismatch" << endl;
                    gotMessage = false;
                } else {
                    // wait for another message is the serial number is wrong.
                    if (msg.serial() != getRespMsg.serial()){
                        cerr << "Serial Numnbers Mismatch" << endl;
                        gotMessage = false;
                    }
                }
            }
        }
    } while(!gotMessage);
    
    if (getRespMsg.has_getres()){
        retVal.status = getRespMsg.getres().status();
        retVal.vlen = getRespMsg.getres().value().size();
        retVal.value = new uint8_t[retVal.vlen];
        memcpy(retVal.value,(uint8_t*)getRespMsg.getres().value().data(), retVal.vlen);
      
    } else {
      cerr << "wrong message type" << endl;
	retVal.status=false;
    }
    cout << "leaving get stub" << endl;
    return retVal;
}

// set up socket for calls to server
bool KVServiceStub::init(){

    // cerr << "client looking for server: " << this->serverName << endl;

    if (this->serverName.empty()){
        cerr << "No server name given -- Does not know server name!! " << endl;
        return false;
    }

    searchRecord = sdstub->searchForService(this->serverName);

    // cerr << "client stub init() port and server from search " << searchRecord.server << " " << searchRecord.port << endl;

    if(searchRecord.port == 0) {
        cerr << "server name not found in directory" << endl;
        return false;
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(searchRecord.port);

    // look up address by name (Version 2 of network.cpp)
    struct addrinfo * res;
    int numAddr = getaddrinfo(searchRecord.server.c_str(), nullptr, nullptr, &res);
    // cerr << "numAddr: " << numAddr << endl;
    // cerr << "number of address results is " << numAddr << endl;
    servaddr.sin_addr = ((struct sockaddr_in*)res -> ai_addr) -> sin_addr;
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

void KVServiceStub::shutdown(){
    if (!ready) return;
    close(sockfd);
    ready = false;
}

