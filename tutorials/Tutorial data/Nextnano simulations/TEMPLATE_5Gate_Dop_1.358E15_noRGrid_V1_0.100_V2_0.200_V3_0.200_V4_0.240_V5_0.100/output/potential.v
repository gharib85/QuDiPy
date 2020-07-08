APPS.SingleWindowApp SingleWindowApp<NEdisplayMode="maximized"> {
   MODS.Read_Field readfield {
      read_field_ui {
         filename = ".\\potential.fld";
         portable = 0;
      };
   };
   MODS.extract_component select {
      in_field => <-.readfield.field;
   };
   MODS.isosurface output {
      in_field => <-.select.out_fld;
   };
   GEOMS.Axis3D axis {
      in_field => <-.select.out_fld;
      x_axis_param {
         axis_name = "x (nm)";
      };
      y_axis_param {
         axis_name = "y (nm)";
      };
      z_axis_param {
         axis_name = "z (nm)";
      };
   };
   GEOMS.TextTitle title {
      TextUI {
         String {
            text = "Potential";
         };
      };
   };
   GDM.Uviewer3D viewer {
      Scene {
         Top {
            child_objs => {
               <-.<-.<-.output.out_obj,
               <-.<-.<-.axis.out_obj,
               <-.<-.<-.title.DefaultObject
            };
         };
      };
   };
};
