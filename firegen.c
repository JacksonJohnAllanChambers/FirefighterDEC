#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>

#define NUM_FIRES 5 //5 fires
#define FIRE_SIZE 8 //each of the 5 fires is 8 tiles long
#define NUM_CITIZENS 500 //number of citizens in NS
#define XMAX 550
#define YMAX 100
#define MIN 0
#define NUM_RNDS 10 //number of rounds may change
#define NUM_FIREFIGHTERS 25


struct fire_data {
	int fire_severity;
	int windspeed;
	int winddir;
	int citizen;
	int firefighter;
}; typedef struct fire_data fire_data;

struct firefighter_coords {
	int x;
	int y;
}; typedef struct firefighter_coords firefighter_coords;

fire_data map[XMAX][YMAX];


int getrand(int min, int max) {
	int randnum = rand() % (max - min + 1) + min;
	return randnum;
}

void print_map(FILE * fout, int round_num, fire_data map[XMAX][YMAX]) {
	fprintf(fout, "%d\n", round_num); //print the round number on the first line of the file
	for (int x = 0; x < XMAX; x++) {
		for (int y = 0; y < YMAX; y++) {
			fprintf(fout, "%03d%02d%d%d", x, y, map[x][y].fire_severity, map[x][y].windspeed);
			fprintf(fout, "%d%d%d\n", map[x][y].winddir, map[x][y].citizen, map[x][y].firefighter);
		}
	}
	fclose(fout);
}


int main(void) {
	FILE* fout = fopen("map.txt", "w");
	unsigned int seed;
	int x, y, x0, y0; //position variables
	int xr_min, xr_max, yr_min, yr_max;
	int round_num = 0; //which round we are on
	int new_round_num = 0; //round listed at top of file
	long mapline; //line read from file
	int cur_severity, new_severity;
	firefighter_coords firefighters[NUM_FIREFIGHTERS];

	//get seed from user
	printf("Enter seed: ");
	scanf("%u", &seed);

	//initialize map
	srand(seed);
	for (x = 0; x < XMAX; x++) {
		for (y = 0; y < YMAX; y++) {
			map[x][y].fire_severity = 0;
			map[x][y].windspeed = getrand(0, 9);
			map[x][y].winddir = getrand(0, 3);
			map[x][y].citizen = 0;
			map[x][y].firefighter = 0;
		}
	}

	//initial fire generation
	for (int i = 0; i < NUM_FIRES; i++) {
		//first fire tile
		x0 = getrand(MIN, XMAX);
		y0 = getrand(MIN, YMAX);
		map[x0][y0].fire_severity = getrand(1, 9);
		//printf("fire at: %d, %d\n", x0, y0);

		//next 9 fire tiles will be within 5 tiles of the first fire
		xr_min = x0 - 5;
		xr_max = x0 + 5;
		yr_min = y0 - 5;
		yr_max = y0 + 5;

		if (xr_min < 0) xr_min = 0;
		if (xr_max > XMAX) xr_max = XMAX;
		if (yr_min < 0) yr_min = 0;
		if (yr_max > YMAX) yr_max = YMAX;
		
		for (int j = 0; j < FIRE_SIZE; j++) {
			x = getrand(xr_min, xr_max);
			y = getrand(yr_min, yr_max);
			map[x][y].fire_severity = getrand(1, 9);
		}

		//distribute firefighters to each fire
		for (int f = 0; f < (NUM_FIREFIGHTERS / NUM_FIRES); f++) {
			x = getrand(xr_min, xr_max);
			y = getrand(yr_min, yr_max);
			if (map[x][y].fire_severity == 0) {
				map[x][y].firefighter = 1; //add a firefighter close to the fire

				//store firefighter positions
				firefighters[f + i*NUM_FIRES].x = x;
				firefighters[f + i*NUM_FIRES].y = y;
			}
			else {
				f--; //can't add a firefighter on top of a fire
			}
		}
	}

	//initial citizen generation
	for (int i = 0; i < NUM_CITIZENS; i++) {
		x = getrand(xr_min, xr_max);
		y = getrand(yr_min, yr_max);
		map[x][y].citizen = 1;
	}

	printf("Round %d\n", round_num);
	
	//print initial fire data to file
	print_map(fout, round_num, map);
	round_num++;
	
	/* ---------- next round ---------- */
	while(round_num < NUM_RNDS){
		//loop until round number changes in file
		fout = fopen("map.txt", "r");
		while (new_round_num != round_num) {
			fseek(fout, 0, SEEK_SET); //go to first line
			fscanf(fout, "%d", &new_round_num);
		}
		fclose(fout);

		printf("Round %d\n", round_num);

		//read updated data from file
		fout = fopen("map.txt", "r");
		fscanf(fout, "%d", &round_num); // store the round number
		for (x = 0; x < XMAX; x++) {
			for (y = 0; y < YMAX; y++) {
				fscanf(fout, "%d", &mapline);
				map[x][y].fire_severity = mapline && 0b10000;
				map[x][y].windspeed = mapline && 0b01000;
				map[x][y].winddir = mapline && 0b00100;
				map[x][y].citizen = mapline && 0b00010;
				map[x][y].firefighter = mapline && 0b00001;
			}
		}
		fclose(fout);

		//spread the fire
		fout = fopen("map.txt", "r+");
		for (x = 0; x < XMAX; x++) {
			for (y = 0; y < YMAX; y++) {
				if (map[x][y].fire_severity > 0) {
					new_severity = map[x][y].fire_severity / 2;
					switch (map[x][y].winddir) {
					case 0:
						y0 = y + 1;
						if (y0 > YMAX) break;
						cur_severity = map[x][y0].fire_severity;
						if (cur_severity < new_severity) {
							map[x][y0].fire_severity = new_severity;
						}
						break;
					case 1:
						x0 = x + 1;
						if (x0 > XMAX) break;
						cur_severity = map[x0][y].fire_severity;
						if (cur_severity < new_severity) {
							map[x0][y].fire_severity = new_severity;
						}
						break;
					case 2:
						y0 = y - 1;
						if (y0 < 0) break;
						cur_severity = map[x][y0].fire_severity;
						if (cur_severity < new_severity) {
							map[x][y0].fire_severity = new_severity;
						}
						break;
					case 3:
						x0 = x - 1;
						if (x0 < 0) break;
						cur_severity = map[x0][y].fire_severity;
						if (cur_severity < new_severity) {
							map[x0][y].fire_severity = new_severity;
						}
						break;
					}

				}
			}
		}

		//fight the fire (firefighters)
		for (int f = 0; f < NUM_FIREFIGHTERS; f++) {
			x0 = firefighters[f].x;
			y0 = firefighters[f].y;
			xr_min = x0 - 1;
			xr_max = x0 + 1;
			yr_min = y0 - 1;
			yr_max = y0 + 1;

			if (map[xr_min][y0].fire_severity > 0) {
				map[xr_min][y0].fire_severity--;
			}
				
			if(map[xr_max][y0].fire_severity > 0) {
				map[xr_max][y0].fire_severity--;
			}
				
			if (map[x0][yr_min].fire_severity > 0) {
				map[x0][yr_min].fire_severity--;
			}
				
			if (map[x0][yr_max].fire_severity > 0) {
				map[x0][yr_max].fire_severity--;
			}
		}

		//update map
		print_map(fout, round_num, map);
		round_num++;
	}

	return 0;
}