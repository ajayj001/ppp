/*
 * asp-par.c:
 * 	All-pairs shortest path implementation based on Floyd's
 * 	algorithms.
 *
 *      Parallel version.
 *
 * Written by Jan van der Lugt, 20-dec-2012
 * Based on a sequential version by Rob van Nieuwpoort, 6-oct-2003
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "mpi.h"

#define MAX_DISTANCE 256
// #define VERBOSE

void alloc_tab(int rows, int cols, int **tab_buf_ptr, int ***tab_ptr) {
	int *tab_buf;
	int **tab;
	int i;

	// malloc the full table in one contiguous block
	tab_buf = (int*)malloc(rows * cols * sizeof(int));
	if (tab_buf == (int *)0) {
		fprintf(stderr, "cannot malloc distance table\n");
		exit(1);
	}

	// create the pointer array
	tab = (int**)malloc(rows * sizeof(int*));
	if (tab == (int **)0) {
		fprintf(stderr, "malloc failed\n");
		exit(1);
	}

	// create a pointer for each row
	for (i = 0; i < rows; i++) {
		tab[i] = tab_buf + i * cols;
	}

	*tab_ptr = tab;
	*tab_buf_ptr = tab_buf;
}

/* malloc and initialize the table with some random distances
 * we never use srand() so rand() will always use the same seed
 * and will hence yields reproducible results (for timing purposes)
 *
 * random initialization of the matrix
 * to be used only for testing purposes
 */
void init_tab(int n, int *mptr, int **tab_buf_ptr, int ***tab_ptr, int oriented) {
	int *tab_buf;
	int **tab;
	int i, j, m = n*n;

	alloc_tab(n, n, &tab_buf, &tab);

	for (i = 0; i < n; i++) {
		tab[i][i] = 0;
		for (j = 0; j < i; j++) {
			tab[i][j] = 1 + (int)((double)MAX_DISTANCE * rand() / (RAND_MAX + 1.0));
			if (oriented) {
				tab[j][i] = 1 + (int)((double)MAX_DISTANCE * rand() / (RAND_MAX + 1.0));
			} else {
				tab[j][i] = tab[i][j];
			}
			if (tab[i][j] == MAX_DISTANCE) m--;
			if (tab[j][i] == MAX_DISTANCE) m--;
		}
	}
	*mptr = m;
	*tab_buf_ptr = tab_buf;
	*tab_ptr = tab;
}


/* reading the list of edges from a file and constructing an adjacency matrix
 * note that it is not mandatory for the graph to be stored as an adjacency matrix -
 * other data structures are allowed, and should be chosen depending on the chosen
 * implementation for the APSP algorithm.
 *
 * The file has the following for mat:
 * first line: [number of vertices] [number of edges] [oriented(0/1)]
 * following [number of edges lines]: [source_node] [destination_node] [weight]
 */
int read_tab(char *INPUTFILE, int *n_ptr, int *m_ptr, int **tab_buf_ptr, int ***tab_ptr, int *o_ptr) {
	/*
	INPUTFILE = name of the graph file
	nptr = number of vertices, to be read from the file
	mptr = number of edges, to be read from the file
	tabptr = the adjancecy matrix for the graph
	optr = returns 1 when the graph is oriented, and 0 otherwise.

	returns: the number of edges that are "incorrect" in the file. That is, in case
	the graph is not oriented, but there are different entries for symmetrical pairs
	of edges, the second such edge is ignored, yet counted for statistics reasons.
	E.g.:

	1 5 20
	5 1 50

	-> If the graph is oriented, these entries are both copied to the adjancency matrix:
	A[1][5] = 20, A[5][1] = 50
	-> If the graph is NOT oriented, the first entry is copied for both pairs, and the second
	one is discarded: A[1][5] = A[5][1] = 20 ; this is a case for an incorrect edge.

	NOTE: the scanning of the file is depenedent on the implementation and the chosen
	data structure for the application. However, it has to be the same for both the sequential
	and the parallel implementations. For the parallel implementation, the file is read by a
	single node, and then distributed to the rest of the participating nodes.
	File reading and graph constructions should not be considered for any timing results.
	*/

	int *tab_buf;
	int **tab;
	int i, j, n, m;
	int source, destination, weight;
	FILE* fp;
	int bad_edges = 0, oriented = 0;

	fp = fopen(INPUTFILE, "r");
	fscanf(fp, "%d %d %d \n", &n, &m, &oriented);
#ifdef VERBOSE
	printf("%d %d %d\n", n, m, oriented);
#endif

	alloc_tab(n, n, &tab_buf, &tab);

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			tab[i][j] = (i == j) ? 0 : MAX_DISTANCE;
		}
	}

	while (!feof(fp)) {
		fscanf(fp, "%d %d %d \n", &source, &destination, &weight);
		if (!oriented) {
			if (tab[source - 1][destination - 1] < MAX_DISTANCE)
				bad_edges++;
			else {
				tab[source - 1][destination - 1] = weight;
				tab[destination - 1][source - 1] = weight;
			}
		} else {
			tab[source - 1][destination - 1] = weight;
		}
	}
	fclose(fp);
#ifdef VERBOSE
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			printf("%5d", tab[i][j]);
		}
		printf("\n");
	}
#endif

	*n_ptr = n;
	*m_ptr = m;
	*tab_buf_ptr = tab_buf;
	*tab_ptr = tab;
	*o_ptr = oriented;
	return bad_edges;
}

void free_tab(int **tab, int n) {
	int i;

	for (i = 0; i < n; i++) {
		free(tab[i]);
	}
	free(tab);
}

void print_tab(int **tab, int n) {
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			printf("%2d ", tab[i][j]);
		}
		printf("\n");
	}
}

int lb_for_node(int n, int rank, int size) {
	float rows_per_node = (float) n / size;
	return floor(rank * rows_per_node);
}

int ub_for_node(int n, int rank, int size) {
	if (rank == size - 1) {
		return n;
	}
	float rows_per_node = (float) n / size;
	return floor((rank + 1) * rows_per_node);
}

int owner_of_row(int k, int n, int size) {
	return (size * (k + 1) - 1) / n;
}

void do_asp(int **tab, int n, int lb, int ub, int rank, int size, MPI_Datatype row_type) {
	int i, j, k, tmp, owner, ret;
	int *pivot_row;
	
	// allocate memory for a temporary row
	pivot_row = (int*)malloc(n * sizeof(int));

	// for each row as a pivot
	for (k = 0; k < n; k++) {
		owner = owner_of_row(k, n, size);
		if (rank == owner) {
			memcpy(pivot_row, tab[k - lb], n * sizeof(int));
		}
		ret = MPI_Bcast(pivot_row, 1, row_type, owner, MPI_COMM_WORLD);
		if (ret != MPI_SUCCESS) {
			printf ("Error scattering data. Terminating.\n");
			MPI_Abort(MPI_COMM_WORLD, ret);
		}
		// proccess every row we own
		for (i = lb; i < ub; i++) {
			if (i != k) {
				for (j = 0; j < n; j++) {
					tmp = tab[i - lb][k] + pivot_row[j];
					if (tmp < tab[i - lb][j]) {
						tab[i - lb][j] = tmp;
					}
				}
			}
		}
	}
	
	free(pivot_row);
}

long int get_sum(int **tab, int n, int m) {
	int i, j;
	long int sum = 0;

	// Get total distance of roads
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (tab[i][j] != MAX_DISTANCE) {
				sum += tab[i][j];
			}
		}
	}
	
	return sum;
}

int get_max(int **tab, int n, int m) {
	int i, j, max = 0;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (tab[i][j] > max) {
				max = tab[i][j];
			}
		}
	}
	
	return max;
}

/******************** Main program *************************/

void usage() {
	printf ("Run the asp program with the following parameters. \n");
	printf (" -read filename :: reads the graph from a file.\n");
	printf (" -random N 0/1 :: generates a NxN graph, randomly. \n");
	printf ("               :: if 1, the graph is oriented, otherwise it is not oriented\n");
	return ;
}

int main(int argc, char *argv[]) {
	int rank, size, ret;
	double starttime = 0.0, endtime = 0.0;

	int i, n, m, lb, ub;
	int max, my_max = 0;
	long int sum, my_sum;
	int bad_edges = 0, oriented = 0;
	int **tab, **my_tab;
	int *tab_buf, *my_tab_buf;
	int print = 0;
	char FILENAME[100];

	MPI_Datatype row_type;
	int *counts, *displs;

	// initialize MPI.
	ret  = MPI_Init(&argc, &argv);
	ret |= MPI_Comm_size(MPI_COMM_WORLD, &size);
	ret |= MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (ret != MPI_SUCCESS) {
		perror("Error with initializing MPI");
		MPI_Abort(MPI_COMM_WORLD, ret);
	}

	// root reads data + prints welcome
	if (rank == 0) {
		usage();

		// read arguments
		n = 0;
		for (i = 1; i < argc; i++) {
			if (!strcmp(argv[i], "-print")) {
				print = 1;
			} else {
				if (!strcmp(argv[i], "-read")) {
					strcpy(FILENAME, argv[i+1]); i++;
				} else {
					if (!strcmp(argv[i], "-random")) {
						n = atoi(argv[i+1]);
						oriented = atoi(argv[i+2]); i += 2;
					} else {
						n = 4000;
						oriented = 1;
					}
				}
			}
		}

		// generate or read graph
		if (n > 0) {
			init_tab(n, &m, &tab_buf, &tab, oriented);
		} else {
			bad_edges = read_tab(FILENAME, &n, &m, &tab_buf, &tab, &oriented);
		}

		fprintf(stderr, "Running ASP with %d rows and %d edges (%d are bad)\n", n, m, bad_edges);
	}

	// broadcast n to all processes
	ret = MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (ret != MPI_SUCCESS) {
		printf ("Error scattering data. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, ret);
	}

	// define MPI datatypes
	ret  = MPI_Type_contiguous(n, MPI_INT, &row_type);
	ret |= MPI_Type_commit(&row_type);
	if (ret != MPI_SUCCESS) {
		printf ("Error defining row type. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, ret);
	}

	// define lower (inclusive) and upper (exclusive) bounds
	lb = lb_for_node(n, rank, size);
	ub = ub_for_node(n, rank, size);

	// allocate buffer for graph on non-root processes
	if (rank != 0) {
		alloc_tab(ub - lb, n, &my_tab_buf, &my_tab);
	}

	// scatter rows to non-root nodes
	counts = (int*)malloc(size * sizeof(int));
	displs = (int*)malloc(size * sizeof(int));
	// send nothing to root
	counts[0] = 0;
	displs[0] = 0;
	for (i = 1; i < size; i++) {
		counts[i] = ub_for_node(n, i, size) - lb_for_node(n, i, size);
		displs[i] = lb_for_node(n, i, size);
	}
	ret = MPI_Scatterv(tab_buf, counts, displs, row_type, my_tab_buf, counts[rank], row_type, 0, MPI_COMM_WORLD);
	if (ret != MPI_SUCCESS) {
		printf ("Error scattering data. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, ret);
	}

	// for root, use main tab as my_tab
	if (rank == 0) {
		my_tab_buf = tab_buf;
		my_tab = tab;
	}

	// root record the start time
	if (rank == 0) {
		starttime = MPI_Wtime();
	}

	// Get the total distance of all roads
	my_sum = get_sum(my_tab, n, ub - lb);
	ret = MPI_Reduce(&my_sum, &sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (ret != MPI_SUCCESS) {
		printf ("Error reducing value. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, ret);
	}

	// Calculate shortest paths between all nodes in the graph
	do_asp(my_tab, n, lb, ub, rank, size, row_type);
	
	// Get maximum distance between any two places
	my_max = get_max(my_tab, n, ub - lb);
	ret = MPI_Reduce(&my_max, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (ret != MPI_SUCCESS) {
		printf ("Error reducing value. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, ret);
	}

	// make sure all processes are finished
	MPI_Barrier(MPI_COMM_WORLD);

	// root records the end time and reports the run time and other statistics
	if (rank == 0) {
		endtime = MPI_Wtime();
		fprintf(stderr, "ASP took %10.3f seconds\n", endtime - starttime);
		printf("Total distance: %ld\n", sum);
		printf("Diameter: %d\n", max);
	}

	// gather rows from non-root rows
	ret = MPI_Gatherv(my_tab_buf, counts[rank], row_type, tab_buf, counts, displs, row_type, 0, MPI_COMM_WORLD);
	if (ret != MPI_SUCCESS) {
		printf ("Error gathering data. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, ret);
	}

	// print the table
	if (rank == 0 && print == 1) {
		print_tab(tab, n);
	}

	// free buffers
	if (rank == 0) {
		free(tab_buf);
		free(tab);
	} else {
		free(my_tab_buf);
		free(my_tab);
	}
	free(displs);
	free(counts);

	// shut down MPI
	ret = MPI_Finalize();

	return 0;
}
