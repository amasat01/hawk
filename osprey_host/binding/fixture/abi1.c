/* A backend library whose entry table carries the abi number 1: the entry of ob_open must refuse it
 * by name. */
#include <stdint.h>
typedef struct {
    uint32_t abi;
} table;
static const table old_table = { 1 };
const table* osprey_v1(void)
{
    return &old_table;
}
