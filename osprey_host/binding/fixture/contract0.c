/* A backend library whose entry table carries the right abi number and another contract version:
 * the entry of ob_open must refuse it by name. */
#include "osprey_host.h"
static const ob_table other = { .abi = OB_ABI, .contract = OB_CONTRACT + 1 };
const ob_table* osprey_v1(void)
{
    return &other;
}
