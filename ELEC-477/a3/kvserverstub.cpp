#include "kvserverstub.hpp"
#include <arpa/inet.h> // For inet_pton
#include <sys/socket.h>
#include <netinet/in.h>


using namespace std::string_literals;
using namespace std;


#define close mclose
void mclose(int fd);

KVServerStub::KVServerStub(string name, string backupIP, in_port_t backupPort){
    //cout << "Entering KVServerStub Constructor" << endl;
    cout << "KVServerStub: Creating KVServerStub for backup: " << backupIP << " on port " << backupPort << endl;
    this->name = name;
    this->backupIP = backupIP;
    this->backupPort = backupPort;
    this->ready = false;
    this->serial = 1;
    cout << "KVServerStub: **BACKUP copy complete** - exiting KVServerStub Constructor" << endl;
}

KVServerStub::KVServerStub(const KVServerStub& kvserverstub){
    //cout << "Entering KVServerStub Copy Constructor" << endl;
    this->name = kvserverstub.name;
    this->backupIP = kvserverstub.backupIP;
    this->backupPort = kvserverstub.backupPort;
    this->ready = kvserverstub.ready;
    this->serial = 1;
    this->sockfd = kvserverstub.sockfd;
    this->servaddr = kvserverstub.servaddr;
    //cout << "Exiting KVServerStub Copy Constructor" << endl;
}



bool KVServerStub::init()
{
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(backupPort);

    struct addrinfo *res;
    int numAddr = getaddrinfo(backupIP.c_str(), nullptr, nullptr, &res);
    // cerr << "number of address results is " << numAddr << endl;
    servaddr.sin_addr = ((struct sockaddr_in *)res->ai_addr)->sin_addr;
    freeaddrinfo(res);

    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
        perror("Socket creation failed");
        return false;
    }

    struct timeval tv;
    tv.tv_sec = 1;  
    tv.tv_usec = 0;  
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv)) < 0)
    {
        perror("error");
        return false;
    }

    cout << "KVServerStub: Socket setup successful" << endl;
    ready = true;
    return true;
}



bool KVServerStub::kvPut(int32_t key, const uint8_t *value, uint16_t vlen)
{
    cout << "[Replica Server] Received PUT request." << endl
         << "[Replica Server] Key: " << key 
         << ", Value Length: " << vlen 
         << ", Value Snippet: '" << string(reinterpret_cast<const char*>(value), min(vlen, static_cast<uint16_t>(10))) << "'..." << endl;
    //cout << "Entering kvPut in KVServerStub" << endl;
    if (!ready) {
    //cout << "im here!" << endl;
    if (!init()) {
        //cout << "im here 2!" << endl;
        return false;
    }
}

    //cout << "Entering kvPut in KVServerStub" << endl;
    cout << "KVServerStub: kvPut called with key: " << key << ", vlen: " << vlen << endl;


    int n;
    socklen_t len;
    uint32_t blen = MAXMSG;
    uint8_t buffer[MAXMSG]; // to serialize into
    struct sockaddr_in servaddrreply;

    // get the current value of serial for this request.
    uint32_t serial = this->serial++;

    // marshal parameters to send.
    E477KV::kvRequest msg;
    msg.set_magic(magic);
    msg.set_version(version1x);
    msg.set_serial(serial++);
    E477KV::putRequest *pr = msg.mutable_putargs();
    pr->set_key(key);
    pr->set_value(std::string((const char *)value, vlen));
    blen = msg.ByteSizeLong();
    if (blen > MAXMSG)
    {
        // too long??
        std::cerr << " *** msg too long" << std::endl;
        // errno = ???
        return false;
    }
    msg.SerializeToArray(buffer, blen);
    n = sendto(sockfd, (const char *)buffer, blen,
               MSG_CONFIRM, (const struct sockaddr *)&servaddr, sizeof(servaddr));


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
    if (putRespMsg.has_putres())
    {
        returnRes = putRespMsg.putres().status();
    }
    else
    {
        cerr << "wrong message type: not put result" << endl;
        returnRes = false;
    }

    //cout << "Exiting kvPut in KVServerStub" << endl;
    cout << "[Replica Server] PUT request processing result: " << (returnRes ? "Success" : "Failure") << endl;

    return returnRes;
}




void KVServerStub::shutdown()
{
    if (!ready)
        return;
    close(sockfd);
    ready = false;
}
