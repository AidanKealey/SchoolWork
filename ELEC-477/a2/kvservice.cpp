
#include "kvservice.hpp"


#ifdef __APPLE__
#define MSG_CONFIRM 0
#endif

using namespace std;
using namespace string_literals;

#define close mclose
void mclose(int fd);

void KVServiceServer::stop(){
    alive = false;
}

void KVServiceServer::start(){
    //cerr << "in kvserviceServer::start" << endl;
    struct sockaddr_in servaddr, cliaddr;

    if(registerService()) {
        cerr << "service successfully registered in directory" << endl;
    } else {
        cerr << "service failed to register in directory" << endl;
    }
    
    // open the GDBM file.
    if (DBMFileName.empty()){
       cerr << "Name of DB file not specified" << endl;
       return;
    }
#ifdef __APPLE__
    dataFile = dbm_open(DBMFileName.c_str(), (O_RDWR | O_CREAT), 0644);
    if (!dataFile){
        cerr << "NDBM Error: could not open database file" << endl;
        return;
    }
#else
    dataFile = gdbm_open(DBMFileName.c_str(), 0, GDBM_WRCREAT, 0644, NULL);
    if (!dataFile){
        cerr << "GDBMM Error: " << gdbm_strerror(gdbm_errno) << endl;
        return;
    }
#endif
    
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
    servaddr.sin_family    = AF_INET;        // IPv4
    servaddr.sin_addr.s_addr = INADDR_ANY;   // whatever interface is available
    servaddr.sin_port = htons(PORT);

    // Bind the socket with the server address
    if (::bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0 )
    {
        perror("bind failed");
        return; // this will exit the service thread and stop the server
    }

    // register here -TD

    socklen_t len;
    int n;
    //cerr << "alive = " << alive << endl;
    while(alive){ 
	//cerr << "waiting for call from client" << endl;

        // wait for a mesage from a client
	len = sizeof(cliaddr);  //len is value/result
	n = recvfrom(sockfd, (uint8_t *)udpMessage, MAXMSG, 
		    MSG_WAITALL, ( struct sockaddr *) &cliaddr,
		    &len);
        //cerr << "server received " << n << " bytes." << endl;
	//std::cerr << HexDump{udpMessage,(uint32_t)n} << endl;

	E477KV::kvRequest receivedMsg;
	E477KV::kvResponse replyMsg;

	if (!receivedMsg.ParseFromArray(udpMessage,n)){
	    cerr << "Couild not parse message" << endl;
	    // ignore
	}

	//cerr << "message parsed" << endl;

	if ((receivedMsg.magic()) != magic){
            cerr << "service unrecognized message" << endl;
        } else {
	    // start by copying version and serial to reply
	    replyMsg.set_magic(magic);
	    replyMsg.set_version(receivedMsg.version());
	    replyMsg.set_serial(receivedMsg.serial());

	    if ((receivedMsg.version() & 0xFF00)  == version1x){
	       // dispatch version 1.x
	       callMethodVersion1(receivedMsg, replyMsg);
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
	    //cerr << "server sent " << servern << " bytes" << endl;
	}

    }

    close(sockfd);

#ifdef __APPLE__
    dbm_close(dataFile);
    dataFile = nullptr;
#else
    gdbm_close(dataFile);
    dataFile = nullptr;
#endif
}



void KVServiceServer::callMethodVersion1(E477KV::kvRequest &receivedMsg, E477KV::kvResponse &replyMsg){
    if (receivedMsg.has_putargs()){
	stringstream ss;
	ss << "put message requested" << endl;
	cerr << ss.str();
        
        const E477KV::putRequest& preq = receivedMsg.putargs();

        int key = preq.key();
        string valueAsStr = preq.value();
        
        bool putRes = kvPut(key, (uint8_t*)valueAsStr.c_str(), valueAsStr.length());
            
        cerr << "put result is " << putRes << endl;
	E477KV::putResponse *presp = replyMsg.mutable_putres();
	presp->set_status(putRes);
    }
    if (receivedMsg.has_getargs()){
	stringstream ss;
	ss << "get message requested" << endl;
	cerr << ss.str();
        
        const E477KV::getRequest& greq = receivedMsg.getargs();

        int key = greq.key();
        
        kvGetResult result = kvGet(key);

	E477KV::getResponse *gr = replyMsg.mutable_getres();
        gr->set_status(result.status);
        gr->set_value(string((char*)result.value,result.vlen));
    }
}

bool KVServiceServer::kvPut(int key, const uint8_t * value, uint16_t vlen){
    stringstream ss;
    //ss = stringstream();
    //ss << "In kvPut, key is " << key << endl;
    //cerr << ss.str();
    
    
#ifdef __APPLE__
    datum dbkey = { (void*)&key, sizeof(int)};
    datum dbvalue = { (void*)value, vlen };
    bool result = dbm_store(dataFile, dbkey, dbvalue, DBM_REPLACE);
#else
    datum dbkey = { (char*)&key, sizeof(int)};
    datum dbvalue = { (char*)value, vlen };
    bool result = gdbm_store(dataFile, dbkey, dbvalue, GDBM_REPLACE);
#endif
    
    return !result;
}


kvGetResult KVServiceServer::kvGet(int key){
    stringstream ss;
    //ss = stringstream();
    //ss << "in kvGet, key is " << key << endl;
    //cerr << ss.str();
    
    kvGetResult res;
    
#ifdef __APPLE__
    datum dbkey = { (void*)&key, sizeof(int)};
    datum value = dbm_fetch(dataFile, dbkey);
#else
    datum dbkey = { (char*)&key, sizeof(int)};
    datum value  = gdbm_fetch(dataFile, dbkey);
#endif
    if (value.dptr == NULL){
        res.status = false;
        res.value = nullptr;
        res.vlen = 0;
    } else {
        res.status = true;
        res.value = (uint8_t*)value.dptr;
        res.vlen =  value.dsize;
    }
    
    return res;
}


bool KVServiceServer::registerService() {
    cerr << "KVServiceServer registering new service" << endl;

    directoryRecord aRecord;
    aRecord.server = this->nodeName();
    aRecord.port = this->PORT;
    
    return sdstub->registerNewServer(this->nickname, aRecord);
}


bool KVServiceServer::deleteService() {
    cerr << "KVServiceServer deleting service" << endl;

    return sdstub->deleteServer(nickname);
}