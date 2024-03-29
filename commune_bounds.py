from osgeo import gdal, ogr
import geopandas as gpd

COMMUNE_NAME = "Pully"


def main() -> None:
    boundaries = gdal.OpenEx(
        "data/swissBOUNDARIES3D_1_5_LV95_LN02.gpkg", gdal.OF_VECTOR
    )
    print("Layers:")
    for i in range(boundaries.GetLayerCount()):
        print("  " + boundaries.GetLayer(i).GetName())

    print("Commune names:")
    for i in range(boundaries.GetLayer(1).GetFeatureCount()):
        feature = boundaries.GetLayer(1).GetFeature(i)
        if not feature:
            continue

        print("  " + feature.GetField("NAME"))
        if feature.GetField("NAME") == COMMUNE_NAME:
            # save the feature as geopackage
            driver = ogr.GetDriverByName("GPKG")
            commune = driver.CreateDataSource("data/commune.gpkg")
            commune_layer = commune.CreateLayer(
                "commune", srs=feature.GetGeometryRef().GetSpatialReference()
            )
            commune_layer.CreateField(ogr.FieldDefn("NAME", ogr.OFTString))
            new_feature = ogr.Feature(commune_layer.GetLayerDefn())
            new_feature.SetField("NAME", feature.GetField("NAME"))
            new_feature.SetGeometry(feature.GetGeometryRef())
            commune_layer.CreateFeature(new_feature)
            new_feature = None
            commune = None
            break

    # Load the geopackage with geopandas
    commune = gpd.read_file("data/commune.gpkg")
    commune.plot().get_figure().savefig("data/commune.png")


if __name__ == "__main__":
    main()
