#include "rag/vector_database.h"
#include <gtest/gtest.h>

namespace ds
{

class VectorDatabaseTest : public ::testing::Test
{
};

TEST_F(VectorDatabaseTest, CheckCreation)
{
    auto db = vector_store_factory(2);

    EXPECT_NE(db, nullptr);
}

TEST_F(VectorDatabaseTest, CheckThrowsOnAdditionInvalidEmbedding)
{
    auto db = vector_store_factory(2);

    EXPECT_THROW(db->add({Embedding{{1.0, 0.0, 1.4}}}), std::logic_error);
}

TEST_F(VectorDatabaseTest, CheckRetrievingTheExact)
{
    auto db = vector_store_factory(2);

    db->add({
        Embedding{{1.f, 0.f}},
        Embedding{{.9f, .1f}},
        Embedding{{.8f, .2f}},
    });

    auto retrieved = db->retrieve(std::vector<float>({0.9f, 0.1f}), 1);

    ASSERT_EQ(1, retrieved.size());

    EXPECT_EQ(retrieved[0].index, 1);
    EXPECT_EQ(retrieved[0].cosine_similarity, 1.0f);
}

TEST_F(VectorDatabaseTest, CheckRetrievingTheClosest)
{
    auto db = vector_store_factory(2);

    db->add({Embedding{{1.f, 0.f}}, Embedding{{.9f, .1f}}, Embedding{{.8f, .2f}}, Embedding{{.7f, .3f}},
             Embedding{{.6f, .4f}}});

    auto retrieved = db->retrieve(std::vector<float>({0.75f, 0.15f}), 1);

    ASSERT_EQ(1, retrieved.size());
    EXPECT_EQ(retrieved[0].index, 2);
    EXPECT_GE(retrieved[0].cosine_similarity, 0.95f);
}

TEST_F(VectorDatabaseTest, CheckRetrievingTheClosestK)
{
    auto db = vector_store_factory(2);

    // Data for this test case was auto generated from python numpy (notebook in data_prep)
    db->add({
        Embedding{{0.536877, 0.857167}}, // similarity: 0.9221
        Embedding{{0.724262, 0.545856}}, // similarity: 0.7230
        Embedding{{0.174767, 0.821576}}, // similarity: 0.9989
        Embedding{{0.183661, 0.183714}}, // similarity: 0.8122
        Embedding{{0.073484, 0.596520}}, // similarity: 0.9992
        Embedding{{0.654126, 0.850020}}, // similarity: 0.8807
        Embedding{{0.516542, 0.628754}}, // similarity: 0.8651
        Embedding{{0.068895, 0.524529}}, // similarity: 0.9995
        Embedding{{0.645730, 0.757034}}, // similarity: 0.8557
        Embedding{{0.278235, 0.984015}}  // similarity: 0.9936
    });

    auto retrieved = db->retrieve(std::vector<float>({0.122122, 0.745582}), 3);

    ASSERT_EQ(3, retrieved.size());
    EXPECT_EQ(retrieved[0].index, 7);
    EXPECT_NEAR(retrieved[0].cosine_similarity, 0.9995f, 0.0001f);
    EXPECT_EQ(retrieved[1].index, 4);
    EXPECT_NEAR(retrieved[1].cosine_similarity, 0.9992f, 0.0001f);
    EXPECT_EQ(retrieved[2].index, 2);
    EXPECT_NEAR(retrieved[2].cosine_similarity, 0.9989f, 0.0001f);
}

TEST_F(VectorDatabaseTest, CheckL2NormWorksForIndexedData)
{
    auto db = vector_store_factory(2);
    // Data for this test case was auto generated from python numpy (notebook in data_prep)
    // only modification is to scale the
    db->add({
        Embedding{{536.877, 857.167}}, // similarity: 0.9221
        Embedding{{724.262, 545.856}}, // similarity: 0.7230
        Embedding{{174.767, 821.576}}, // similarity: 0.9989
        Embedding{{183.661, 183.714}}, // similarity: 0.8122
        Embedding{{73.484, 596.520}},  // similarity: 0.9992
        Embedding{{654.126, 850.020}}, // similarity: 0.8807
        Embedding{{516.542, 628.754}}, // similarity: 0.8651
        Embedding{{68.895, 524.529}},  // similarity: 0.9995
        Embedding{{645.730, 757.034}}, // similarity: 0.8557
        Embedding{{278.235, 984.015}}  // similarity: 0.9936
    });

    auto retrieved = db->retrieve(std::vector<float>({0.122122, 0.745582}), 1);

    ASSERT_EQ(1, retrieved.size());
    EXPECT_EQ(retrieved[0].index, 7);
    EXPECT_NEAR(retrieved[0].cosine_similarity, .9995f, 0.0001f);
}

TEST_F(VectorDatabaseTest, CheckL2NormWorksForSearchData)
{
    auto db = vector_store_factory(2);

    // Data for this test case was auto generated from python numpy (notebook in data_prep)
    db->add({
        Embedding{{0.536877, 0.857167}}, // similarity: 0.9221
        Embedding{{0.724262, 0.545856}}, // similarity: 0.7230
        Embedding{{0.174767, 0.821576}}, // similarity: 0.9989
        Embedding{{0.183661, 0.183714}}, // similarity: 0.8122
        Embedding{{0.073484, 0.596520}}, // similarity: 0.9992
        Embedding{{0.654126, 0.850020}}, // similarity: 0.8807
        Embedding{{0.516542, 0.628754}}, // similarity: 0.8651
        Embedding{{0.068895, 0.524529}}, // similarity: 0.9995
        Embedding{{0.645730, 0.757034}}, // similarity: 0.8557
        Embedding{{0.278235, 0.984015}}  // similarity: 0.9936
    });
    auto retrieved = db->retrieve(std::vector<float>({122.122, 745.582}), 1);

    ASSERT_EQ(1, retrieved.size());
    EXPECT_EQ(retrieved[0].index, 7);
    EXPECT_NEAR(retrieved[0].cosine_similarity, .9995f, 0.0001f);

    auto retrieved2 = db->retrieve(std::vector<float>({1.22122, 7.45582}), 1);

    ASSERT_EQ(1, retrieved2.size());
    EXPECT_EQ(retrieved2[0].index, 7);
    EXPECT_NEAR(retrieved2[0].cosine_similarity, .9995f, 0.0001f);
}

TEST_F(VectorDatabaseTest, CheckHigherRankkDatabase)
{
    auto db = vector_store_factory(10);

    // Data for this test case was auto generated from python numpy (notebook in data_prep)
    db->add({
        Embedding{{0.158970, 0.042453, 0.879347, 0.300486, 0.751313, 0.295132, 0.139527, 0.719000, 0.252506,
                   0.335722}}, // similarity: 0.6219
        Embedding{{0.186042, 0.421800, 0.269480, 0.514075, 0.813258, 0.226128, 0.599426, 0.435857, 0.414255,
                   0.909104}}, // similarity: 0.7581
        Embedding{{0.751599, 0.389995, 0.427895, 0.850181, 0.550109, 0.672254, 0.384026, 0.473392, 0.325718,
                   0.448030}}, // similarity: 0.8884
        Embedding{{0.617815, 0.513401, 0.599782, 0.920432, 0.673105, 0.953881, 0.179667, 0.970486, 0.401055,
                   0.854085}}, // similarity: 0.8457
        Embedding{{0.973778, 0.499899, 0.652928, 0.933846, 0.739566, 0.138782, 0.154604, 0.543544, 0.027348,
                   0.876642}}, // similarity: 0.7915
        Embedding{{0.416288, 0.479054, 0.825799, 0.705752, 0.584601, 0.730077, 0.825270, 0.866505, 0.505773,
                   0.408417}}, // similarity: 0.8097
        Embedding{{0.614848, 0.652643, 0.635558, 0.288801, 0.005912, 0.409191, 0.565267, 0.738173, 0.849659,
                   0.022323}}, // similarity: 0.7963
        Embedding{{0.698526, 0.138389, 0.314190, 0.700148, 0.216304, 0.817933, 0.046192, 0.553444, 0.093071,
                   0.216965}}, // similarity: 0.7879
        Embedding{{0.438814, 0.088136, 0.507943, 0.758511, 0.067982, 0.256245, 0.823911, 0.248976, 0.003543,
                   0.707839}}, // similarity: 0.7517
        Embedding{{0.339896, 0.423779, 0.721870, 0.702700, 0.479762, 0.414201, 0.630093, 0.608672, 0.264866,
                   0.885349}} // similarity: 0.8300
    });

    auto retrieved = db->retrieve(std::vector<float>({0.918729, 0.525682, 0.579159, 0.479968, 0.209927, 0.759181,
                                                      0.345223, 0.097933, 0.743002, 0.793644}),
                                  1);

    ASSERT_EQ(1, retrieved.size());
    EXPECT_EQ(retrieved[0].index, 2);
    EXPECT_NEAR(retrieved[0].cosine_similarity, .8884f, 0.0001f);
}

TEST_F(VectorDatabaseTest, CheckAddingNewVectorsAffectsResult)
{
    auto db = vector_store_factory(2);

    // Data for this test case was auto generated from python numpy (notebook in data_prep)
    db->add({
        Embedding{{0.536877, 0.857167}}, // similarity: 0.9221
        Embedding{{0.724262, 0.545856}}, // similarity: 0.7230
        Embedding{{0.174767, 0.821576}}, // similarity: 0.9989
        Embedding{{0.183661, 0.183714}}, // similarity: 0.8122
        Embedding{{0.073484, 0.596520}}, // similarity: 0.9992
        Embedding{{0.654126, 0.850020}}, // similarity: 0.8807
        Embedding{{0.516542, 0.628754}}, // similarity: 0.8651
        Embedding{{0.645730, 0.757034}}, // similarity: 0.8557
        Embedding{{0.278235, 0.984015}}  // similarity: 0.9936
    });
    auto retrieved_1 = db->retrieve(std::vector<float>({122.122, 745.582}), 1);

    EXPECT_EQ(retrieved_1[0].index, 4);

    db->add({
        Embedding{{0.068895, 0.524529}}, // similarity: 0.9995
    });

    auto retrieved_2 = db->retrieve(std::vector<float>({122.122, 745.582}), 1);

    EXPECT_EQ(retrieved_2[0].index, 9);
}
} // namespace ds