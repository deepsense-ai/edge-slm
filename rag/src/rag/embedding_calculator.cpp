#include "rag/embedding_calculator.h"

namespace ds
{
EmbeddingCalculationResult IEmbeddingCalculator::calc(const std::string& chunk) const
{
    return calc_batch({chunk})[0];
}
} // namespace ds