//
//  genData.cpp
//  
// generates simulated test daata for assignment4
//
//
//  Created by Thomas Dean on 2024-03-21.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;


//siulation parameters
double updatePeriod = 4.0;
unsigned long long initialTimestamp = 1656331210;
double aiportElev = 173.0; // YYZ
double feetPerMeter = 3.281;

// coordinates in degrees
// altitude in meters
// velocity in m/s

struct Flight {
    double lat1; // start point
    double long1;
    double alt1;
    
    double lat2; // end coordinate
    double long2;
    double alt2;

    double velocity;
    string callsign;
    string icao24;

    
};

// 600 mph = 268.225 m/s
// 370 mph = 165.405 m/s
// 200 mph = 89.408 m/s
// 115 mph = 51.4096 m/s
// 16.5 mph = 7.37616 m/s

// YYZ Location 43.683, -79.633

vector<Flight> flights = {
    //  lat1, long1,      alt1               lat2  long2        alt2                   vel      call     ico24
    { 43.776, -78.8, (30000 / feetPerMeter), 43.44, -80.5,    (30000 / feetPerMeter), 268.225, "KM20", "a1234"},
    { 43.82,  -79.8, (2000 / feetPerMeter),  43.34, -79.5,    (2000 / feetPerMeter),  89.408 , "PT130", "a3456"},
    { 43.75, -79.65, (4400 / feetPerMeter),  43.683, -79.633, aiportElev,             89.408 , "AC180", "a9923"},
    { 43.683, -79.633,      aiportElev,         44,  -79.8,   (30000 / feetPerMeter), 92.0 , "WJ910", "a99874"}
};

char * progName = nullptr;

int main(int argc, char * argv[]){
    
    // Get data file name from command line
    progName = argv[0];
    if (argc != 2){
        cerr << "Usage: " << progName << " dataFileName.csv" << endl;
        exit(1);
    }
    
    const string fileName = argv[1];
    ofstream file(fileName);
    if (file.fail()){
        cerr << progName << ": could not open file " << fileName << " for write" << endl;
        exit(1);
    }
    
    // generate data for each aircraft in turn
    vector<Flight>::iterator it;
    for(it = flights.begin(); it != flights.end(); it++){
        
        cout << "=======" << it-> callsign << "=======" << endl;
        cout << "Starting point (" << it -> lat1 << ", " << it -> long1 << ")" << endl;
        cout << "ending   point (" << it -> lat2 << ", " << it -> long2 << ")" << endl;
        cout << "velocity is " << it-> velocity << endl;
        
        // haversine distance
        const double R = 6371000; // metres
        const double lat1rad = it->lat1 * M_PI/180; // φ, λ in radians
        const double lat2rad = it->lat2 * M_PI/180;
        const double long1rad = it->long1 * M_PI/180;
        const double long2rad = it->long2 * M_PI/180;
        const double deltaLat = (it->lat2 - it->lat1) * M_PI/180;
        const double deltaLong = (it->long2 - it->long1) * M_PI/180;

        const double a = sin(deltaLat/2) * sin(deltaLat/2) +
                  cos(lat1rad) * cos(lat2rad) *
                  sin(deltaLong/2) * sin(deltaLong/2);
        const double c = 2 * atan2(sqrt(a), sqrt(1-a));
        const double d = R * c; // in metres
        
        cout << "haverstine distance is " << d << endl;
        
#ifdef COMP_HAVER_AND_EQU
        // equirectangular projection estimate (to compare)
        // for first test haversine distance was 141878, equirectangular was 141881
        const double ex = (long2rad - long1rad) * cos((lat1rad + lat2rad)/2);
        const double ey = (lat2rad - lat1rad);
        const double d2 = sqrt(ex*ex + ey*ey) * R;
        cout << "equirectangular distance is "  << d2 << endl;
        cout << "difference is " << (fabs(d2-d)/d * 100) << "%" << endl;
#endif
        
        // bearing
        const double y = sin(deltaLong) * cos(lat2rad);
        const double x = cos(lat1rad)* sin(lat2rad) -
                  sin(lat1rad)*cos(lat2rad)*cos(deltaLong);
        const double theta = atan2(y, x);
        double brng = (theta*180/M_PI + 360);
        if (brng > 360.0) brng -= 360.0; // in degrees
        if (brng < -360.0) brng += 360.0;
        cout << "direction is " << brng << endl;
        
        // step and upate values to generate samples
        double distPerStep = it->velocity * updatePeriod;
        int numSteps = (int)(d / distPerStep);
        double totalTime = d / it-> velocity;
        double degreesLatPerStep = (it->lat2 - it->lat1) / numSteps;
        double degreesLongPerStep = (it->long2 - it->long1) / numSteps;
        
        // altitude chagnge over time is diffeerence in altitude / total time
        double vertRate = (it->alt2 - it->alt1) / totalTime;
        double vertPerStep = (it->alt2 - it->alt1) / numSteps;
        
        cout << "total time " << totalTime << endl;
        cout << "meters per step " << distPerStep << endl;
        cout << "numSteps " << numSteps << endl;
        cout << "degreesLatPerStep " << degreesLatPerStep << endl;
        cout << "degreesLlongPerStep " << degreesLongPerStep << endl;
        cout << "vertical Reate " << vertRate << endl;

        // step variables
        unsigned long long timeStamp = initialTimestamp;
        double lat = it->lat1;
        double longitude = it->long1;
        double alt = it->alt1;
        
        for (int i = 0; i < numSteps; i++){
            //time,icao24,lat,lon,velocity,heading,vertrate,callsign,onground,alert,spi,squawk,baroaltitude,geoaltitude,lastposupdate,lastcontact
            file << timeStamp << "," << it->icao24;
            file << "," << lat;
            file << "," << longitude;
            file << "," << it->velocity;
            file << "," << brng;
            file << "," << vertRate;
            file << "," << it->callsign;
            file << ",FALSE,FALSE,FALSE,1200";     // onground, alert, spi, squawk;
            file << "," << alt;     // bareoalt
            file << "," << alt;     // gepalt
            file << "," << timeStamp - (int)(updatePeriod);
            file << "," << timeStamp;
            file << endl;
            // update for next sample
            timeStamp += (int)updatePeriod;
            lat += degreesLatPerStep;
            longitude += degreesLongPerStep;
            alt += vertPerStep;
        }
        
    }
    file.close();
    
}
