#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

// Include the Weld API.
#include "../../c/weld.h"

#define PASS 42

struct weld_vector {
    int32_t *data;
    int64_t length;
};

struct args {
    struct weld_vector in1;
    struct weld_vector in2;
    struct weld_vector in3;
    struct weld_vector in4;
    struct weld_vector in5;
    struct weld_vector in6;
};

char *read_code() {
    FILE *fptr = fopen("pred_vs_no_pred.weld", "r");
    fseek(fptr, 0, SEEK_END);
    int string_size = ftell(fptr);
    rewind(fptr);
    char *program = (char *) malloc(sizeof(char) * (string_size + 1));
    fread(program, sizeof(char), string_size, fptr);
    program[string_size] = '\0';
    return program;
}

struct weld_vector generate_filter_vec(int64_t n, float prob) {
    int pass_thres = 10000000 * prob;

    struct weld_vector result;
    result.data = (int32_t *)malloc(sizeof(int32_t) * n);
    result.length = n;
    for (int i = 0; i < n; i++) {
        if (rand() % 10000000 <= pass_thres) {
            result.data[i] = PASS;
        } else {
            result.data[i] = rand() % PASS;
        }
    }
    return result;
}

struct weld_vector generate_random_vec(int64_t n) {
    struct weld_vector result;
    result.data = (int32_t *)malloc(sizeof(int32_t) * n);
    result.length = n;
    for (int i = 0; i < n; i++) {
        result.data[i] = rand() % 1000;
    }
    return result;
}

struct args generate_data(int64_t n, float prob) {
    struct args result;
    result.in1 = generate_filter_vec(n, prob);
    result.in2 = generate_random_vec(n);
    result.in3 = generate_random_vec(n);
    result.in4 = generate_random_vec(n);
    result.in5 = generate_random_vec(n);
    result.in6 = generate_random_vec(n);
    return result;
}

int main(int argc, char **argv) {
    int32_t n = atoi(argv[1]);
    float prob = atof(argv[2]);

    // Load the code and generate data.
    char *program = read_code();
    struct args a = generate_data(n, prob);
    weld_value_t arg = weld_value_new(&a);

    // Compile Weld module.
    weld_error_t e = weld_error_new();
    weld_conf_t conf = weld_conf_new();
    weld_module_t m = weld_module_compile(program, conf, e);

    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Compile error: %s\n", err);
        exit(1);
    }

    // Run the module and get the result.
    weld_value_t result = weld_module_run(m, conf, arg, e);
    if (weld_error_code(e)) {
        const char *err = weld_error_message(e);
        printf("Runtime error: %s\n", err);
        exit(1);
    }
    void *result_data = weld_value_data(result);
    printf("Answer: %d\n", *(int32_t *)result_data);

    // Free the values.
    weld_value_free(result);
    weld_value_free(arg);
    weld_conf_free(conf);

    weld_error_free(e);
    weld_module_free(m);
    return 0;
}